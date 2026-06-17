# Copyright (c) OpenMMLab. All rights reserved.
"""Process-local manager for EPD encoder-output transfer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.disagg.conn.protocol import (
    EPDInitRequest,
    EncoderCacheFreeRequest,
    EncoderOutputEntry,
    EncoderOutputMetadata,
    EncoderOutputRef,
    EncoderTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.epd.cache import EncoderCache, get_epd_encoder_cache
from lmdeploy.pytorch.disagg.epd.control import EncoderTransferConfig, free_remote_encoder_cache_ref_async
from lmdeploy.pytorch.disagg.epd.dlslime import DLSlimeEndpoint
from lmdeploy.pytorch.messages import InputEmbeddings


@dataclass
class _EmbeddingLayout:
    tensor: torch.Tensor
    ranges: list[list[int]]
    shapes: list[list[int]]
    dtype: str
    nbytes: int


def _embedding_tensor(embedding, device: torch.device) -> torch.Tensor:
    data = getattr(embedding, 'embeddings', embedding)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    else:
        tensor = torch.as_tensor(data)
    return tensor.to(device=device, non_blocking=True).contiguous()


def _build_embedding_layout(prompt_input: dict, device: torch.device) -> _EmbeddingLayout:
    input_embeddings = prompt_input.get('input_embeddings') or []
    if not input_embeddings:
        raise ValueError('DLSlime encoder transfer requires precomputed input_embeddings.')

    input_embedding_ranges = prompt_input.get('input_embedding_ranges')
    tensors: list[torch.Tensor] = []
    ranges: list[list[int]] = []
    shapes: list[list[int]] = []
    dtype: torch.dtype | None = None
    hidden_size: int | None = None

    for index, embedding in enumerate(input_embeddings):
        if input_embedding_ranges is not None:
            start, end = [int(value) for value in input_embedding_ranges[index]]
        elif hasattr(embedding, 'start') and hasattr(embedding, 'end'):
            start, end = int(embedding.start), int(embedding.end)
        else:
            raise ValueError('input_embedding_ranges are required for DLSlime encoder embeddings.')
        tensor = _embedding_tensor(embedding, device)
        if tensor.ndim != 2:
            raise ValueError(f'DLSlime encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
        if end - start != tensor.shape[0]:
            raise ValueError(f'DLSlime encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}')
        if dtype is None:
            dtype = tensor.dtype
            hidden_size = tensor.shape[1]
        elif tensor.dtype != dtype or tensor.shape[1] != hidden_size:
            raise ValueError('DLSlime encoder embeddings must share dtype and hidden size.')
        tensors.append(tensor)
        ranges.append([start, end])
        shapes.append(list(tensor.shape))

    flat_tensor = torch.cat(tensors, dim=0).contiguous()
    return _EmbeddingLayout(
        tensor=flat_tensor,
        ranges=ranges,
        shapes=shapes,
        dtype=str(flat_tensor.dtype).replace('torch.', ''),
        nbytes=flat_tensor.numel() * flat_tensor.element_size(),
    )


class EncoderTransferManager:
    """Owns one process-local encoder-output transfer endpoint."""

    def __init__(self,
                 engine_id: str,
                 *,
                 device: str | torch.device | None = None,
                 endpoint=None,
                 link_type: str = 'RoCE',
                 ib_port: int = 1,
                 rank: int = 0,
                 device_name: str | None = None,
                 encoder_cache: EncoderCache | None = None):
        self.engine_id = engine_id
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.transfer_endpoint = DLSlimeEndpoint(
            endpoint=endpoint,
            link_type=link_type,
            ib_port=ib_port,
            rank=rank,
            device_name=device_name,
        )
        self.endpoint = self.transfer_endpoint.endpoint
        self.encoder_cache = encoder_cache if encoder_cache is not None else get_epd_encoder_cache()
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._published_mr_keys: dict[str, str] = {}
        self._transfer_pins: dict[str, list[str]] = {}

    @property
    def endpoint_info(self):
        return self.transfer_endpoint.endpoint_info

    def p2p_initialize(self, init_request: EPDInitRequest) -> EncoderTransferEndpointInfo:
        return EncoderTransferEndpointInfo(protocol=init_request.protocol, endpoint_info=self.endpoint_info)

    def p2p_connect(self, remote_engine_id: str, endpoint_info: EncoderTransferEndpointInfo):
        self.transfer_endpoint.connect_once(remote_engine_id, endpoint_info.endpoint_info)

    def p2p_drop_connect(self, remote_engine_id: str):
        self.transfer_endpoint.drop_connect(remote_engine_id)

    async def publish(self,
                      prompt_input: dict,
                      remote_engine_id: str,
                      remote_session_id: int,
                      transfer_id: str) -> EncoderOutputRef:
        if not self._can_publish_cached(prompt_input):
            self._cache_prompt_embeddings(prompt_input)
        if self._can_publish_cached(prompt_input):
            return self._publish_cached_entries(prompt_input, remote_engine_id, remote_session_id, transfer_id)
        return self._publish_flat_one_shot(prompt_input, remote_engine_id, remote_session_id, transfer_id)

    def _can_publish_cached(self, prompt_input: dict) -> bool:
        cache = self.encoder_cache
        if cache is None:
            return False
        input_embeddings = prompt_input.get('input_embeddings') or []
        cache_keys = prompt_input.get('epd_encoder_cache_keys') or []
        if len(input_embeddings) == 0 or len(input_embeddings) != len(cache_keys):
            return False
        for cache_key in cache_keys:
            if cache_key is None or cache.get_entry(cache_key) is None:
                return False
        return True

    def _cache_prompt_embeddings(self, prompt_input: dict):
        cache = self.encoder_cache
        if cache is None:
            return
        input_embeddings = prompt_input.get('input_embeddings') or []
        cache_keys = prompt_input.get('epd_encoder_cache_keys') or []
        if len(input_embeddings) == 0 or len(input_embeddings) != len(cache_keys):
            return

        for embedding, cache_key in zip(input_embeddings, cache_keys):
            if cache_key is None or cache.get_entry(cache_key) is not None:
                continue
            start, end = int(embedding.start), int(embedding.end)
            tensor = _embedding_tensor(embedding, self.device)
            if tensor.ndim != 2:
                raise ValueError(f'DLSlime encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
            if end - start != tensor.shape[0]:
                raise ValueError(f'DLSlime encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}')
            try:
                cache.put(cache_key, tensor)
            except RuntimeError:
                return

    def _publish_cached_entries(self, prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                                transfer_id: str) -> EncoderOutputRef:
        input_embeddings = prompt_input.get('input_embeddings') or []
        cache_keys = prompt_input.get('epd_encoder_cache_keys') or []
        entries: list[EncoderOutputEntry] = []
        pinned_keys = []
        assert self.encoder_cache is not None

        try:
            for embedding, cache_key in zip(input_embeddings, cache_keys):
                entry = self.encoder_cache.get_entry(cache_key)
                assert entry is not None
                tensor = entry.tensor
                start, end = int(embedding.start), int(embedding.end)
                if tensor.ndim != 2:
                    raise ValueError(f'DLSlime encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
                if end - start != tensor.shape[0]:
                    raise ValueError(
                        f'DLSlime encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}'
                    )

                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                if entry.mr_key is None:
                    mr_key = f'epd-cache-{cache_key}'
                    self.transfer_endpoint.register_memory_region(mr_key, tensor.data_ptr(), entry.nbytes)
                    entry.mr_key = mr_key

                    def _unregister_cache_entry(mr_key=entry.mr_key):
                        self.transfer_endpoint.unregister_memory_region(mr_key)

                    entry.on_evict = _unregister_cache_entry

                # Keep cached tensor/MR alive until the receiver finishes this transfer.
                self.encoder_cache.pin(cache_key)
                pinned_keys.append(cache_key)
                entries.append(
                    EncoderOutputEntry(
                        cache_key=cache_key,
                        mr_key=entry.mr_key,
                        mr_info=self.transfer_endpoint.mr_info(entry.mr_key),
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype).replace('torch.', ''),
                        nbytes=entry.nbytes,
                        input_embedding_range=[start, end],
                    ))
        except Exception:
            for cache_key in pinned_keys:
                self.encoder_cache.unpin(cache_key)
            raise

        self._transfer_pins[transfer_id] = pinned_keys

        return EncoderOutputRef(
            token_ids=list(prompt_input.get('input_ids') or []),
            input_embedding_ranges=[entry.input_embedding_range for entry in entries],
            protocol=MigrationProtocol.RDMA,
            transfer_id=transfer_id,
            remote_engine_id=remote_engine_id,
            remote_session_id=remote_session_id,
            dtype=entries[0].dtype,
            shape=[entry.shape for entry in entries],
            transfer_metadata=EncoderOutputMetadata(endpoint_info=self.endpoint_info, entries=entries),
        )

    def _publish_flat_one_shot(self, prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                               transfer_id: str) -> EncoderOutputRef:
        layout = _build_embedding_layout(prompt_input, self.device)

        # Encoder output is produced asynchronously on CUDA; wait before registering the
        # RDMA buffer so the receiver never reads data still being written.
        if layout.tensor.is_cuda:
            torch.cuda.synchronize(layout.tensor.device)

        self.transfer_endpoint.register_memory_region(transfer_id, layout.tensor.data_ptr(), layout.nbytes)
        self._published_tensors[transfer_id] = layout.tensor
        self._published_mr_keys[transfer_id] = transfer_id

        return EncoderOutputRef(
            token_ids=list(prompt_input.get('input_ids') or []),
            input_embedding_ranges=layout.ranges,
            protocol=MigrationProtocol.RDMA,
            transfer_id=transfer_id,
            remote_engine_id=remote_engine_id,
            remote_session_id=remote_session_id,
            dtype=layout.dtype,
            shape=layout.shapes,
            transfer_metadata=EncoderOutputMetadata(
                endpoint_info=self.endpoint_info,
                mr_info=self.transfer_endpoint.mr_info(transfer_id),
                nbytes=layout.nbytes,
            ),
        )

    async def receive(self, encoder_output_ref: EncoderOutputRef) -> dict:
        entries = encoder_output_ref.transfer_metadata.entries
        if entries is not None:
            return await self._receive_cached_entries(encoder_output_ref, entries)
        return await self._receive_flat_one_shot(encoder_output_ref)

    async def _receive_cached_entries(self, encoder_output_ref: EncoderOutputRef,
                                      entries: list[EncoderOutputEntry]) -> dict:
        endpoint_info = encoder_output_ref.transfer_metadata.endpoint_info
        if len(entries) != len(encoder_output_ref.input_embedding_ranges):
            raise ValueError('DLSlime encoder_output_ref entry and embedding range counts do not match.')

        self.transfer_endpoint.connect_once(encoder_output_ref.remote_engine_id, endpoint_info)
        embeddings = []
        for entry in entries:
            start, end = entry.input_embedding_range
            shape = entry.shape
            if len(shape) != 2:
                raise ValueError('DLSlime encoder_output_ref requires 2-D embedding shapes.')
            dtype = getattr(torch, entry.dtype)
            output = torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=self.device)
            expected_nbytes = output.numel() * output.element_size()
            nbytes = int(entry.nbytes)
            if nbytes != expected_nbytes:
                raise ValueError('DLSlime encoder_output_ref byte size does not match embedding shape and dtype.')
            if int(end) - int(start) != int(shape[0]):
                raise ValueError('DLSlime encoder_output_ref range length does not match embedding shape.')
            await self.transfer_endpoint.read_into(output, entry.mr_info, nbytes)
            embeddings.append(InputEmbeddings(output, start=int(start), end=int(end)))
        return dict(prompt=None, input_ids=list(encoder_output_ref.token_ids), input_embeddings=embeddings)

    async def _receive_flat_one_shot(self, encoder_output_ref: EncoderOutputRef) -> dict:
        endpoint_info = encoder_output_ref.transfer_metadata.endpoint_info
        remote_mr_info = encoder_output_ref.transfer_metadata.mr_info
        nbytes = encoder_output_ref.transfer_metadata.nbytes
        if endpoint_info is None or remote_mr_info is None or nbytes is None:
            raise ValueError('DLSlime encoder_output_ref is missing endpoint or memory-region metadata.')

        shapes = encoder_output_ref.shape or []
        if not isinstance(shapes, list) or len(shapes) == 0 or not isinstance(shapes[0], list):
            raise ValueError('DLSlime encoder_output_ref requires per-embedding shapes.')
        ranges = encoder_output_ref.input_embedding_ranges
        if len(shapes) != len(ranges):
            raise ValueError('DLSlime encoder_output_ref shape and embedding range counts do not match.')
        dtype = getattr(torch, encoder_output_ref.dtype)
        total_rows = sum(int(shape[0]) for shape in shapes)
        hidden_size = int(shapes[0][1])
        for shape, (start, end) in zip(shapes, ranges):
            if len(shape) != 2:
                raise ValueError('DLSlime encoder_output_ref requires 2-D embedding shapes.')
            if int(shape[1]) != hidden_size:
                raise ValueError('DLSlime encoder_output_ref embeddings must share hidden size.')
            if int(end) - int(start) != int(shape[0]):
                raise ValueError('DLSlime encoder_output_ref range length does not match embedding shape.')
        output = torch.empty((total_rows, hidden_size), dtype=dtype, device=self.device)
        expected_nbytes = output.numel() * output.element_size()
        if int(nbytes) != expected_nbytes:
            raise ValueError('DLSlime encoder_output_ref byte size does not match embedding shape and dtype.')

        self.transfer_endpoint.connect_once(encoder_output_ref.remote_engine_id, endpoint_info)
        await self.transfer_endpoint.read_into(output, remote_mr_info, int(nbytes))

        embeddings = []
        offset = 0
        for shape, (start, end) in zip(shapes, ranges):
            rows = int(shape[0])
            embeddings.append(InputEmbeddings(output[offset:offset + rows], start=int(start), end=int(end)))
            offset += rows
        return dict(prompt=None, input_ids=list(encoder_output_ref.token_ids), input_embeddings=embeddings)

    def release_published(self, transfer_id: str):
        for cache_key in self._transfer_pins.pop(transfer_id, []):
            if self.encoder_cache is not None:
                # Release only this transfer ref; the cache entry remains reusable.
                self.encoder_cache.unpin(cache_key)
        key = self._published_mr_keys.pop(transfer_id, None)
        if key is not None:
            self.transfer_endpoint.unregister_memory_region(key)
        self._published_tensors.pop(transfer_id, None)

    def close(self):
        for transfer_id in list(self._published_tensors):
            self.release_published(transfer_id)
        if self.encoder_cache is not None:
            self.encoder_cache.clear()
        self.transfer_endpoint.close()


_ENCODER_TRANSFER_MANAGER: EncoderTransferManager | None = None


def set_encoder_transfer_manager(manager: EncoderTransferManager | None):
    global _ENCODER_TRANSFER_MANAGER
    _ENCODER_TRANSFER_MANAGER = manager


def get_encoder_transfer_manager() -> EncoderTransferManager:
    if _ENCODER_TRANSFER_MANAGER is None:
        raise ValueError('EPD transfer manager is not initialized.')
    return _ENCODER_TRANSFER_MANAGER


async def publish_encoder_output(prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                                 transfer_config: EncoderTransferConfig) -> EncoderOutputRef:
    """Publish computed encoder output."""
    manager = get_encoder_transfer_manager()
    return await manager.publish(prompt_input, remote_engine_id, remote_session_id, transfer_config.transfer_id)


async def load_encoder_output_async(encoder_output_ref: EncoderOutputRef) -> dict:
    """Convert an encoder-output ref into prompt input."""
    manager = get_encoder_transfer_manager()
    try:
        return await manager.receive(encoder_output_ref)
    finally:
        asyncio.create_task(free_remote_encoder_cache_ref_async(encoder_output_ref))


async def free_published_encoder_cache_ref_async(request: EncoderCacheFreeRequest) -> None:
    """Free producer-side encoder cache reference state."""
    manager = get_encoder_transfer_manager()
    manager.release_published(request.transfer_id)
