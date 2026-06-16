# Copyright (c) OpenMMLab. All rights reserved.
"""DLSlime/RDMA manager for EPD encoder-output transfer."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch

from lmdeploy.pytorch.disagg.conn.protocol import (
    EncoderCacheFreeRequest,
    EncoderOutputEntry,
    EncoderOutputMetadata,
    EncoderOutputRef,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.epd.cache import EncoderCache, get_epd_encoder_cache
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

@dataclass(frozen=True)
class EncoderTransferConfig:
    """Per-request encoder transfer configuration."""

    transfer_id: str | None = None
    receiver_endpoint_info: dict | None = None
    receiver_engine_id: str | None = None

    def __post_init__(self):
        if not self.transfer_id:
            raise ValueError('EPD DLSlime RDMA transfer requires transfer_id')
        if not self.receiver_endpoint_info:
            raise ValueError('EPD DLSlime RDMA transfer requires receiver_endpoint_info')

    @classmethod
    def from_request(cls, request_dict: dict) -> 'EncoderTransferConfig':
        """Build transfer config from internal request fields."""
        return cls(
            transfer_id=request_dict.get('epd_transfer_id'),
            receiver_endpoint_info=request_dict.get('encoder_output_receiver_endpoint_info'),
            receiver_engine_id=request_dict.get('encoder_output_receiver_engine_id'),
        )

    def to_request_fields(self) -> dict:
        """Serialize transfer config into internal request fields."""
        fields = {'epd_transfer_id': self.transfer_id}
        if self.receiver_endpoint_info:
            fields['encoder_output_receiver_endpoint_info'] = self.receiver_endpoint_info
        if self.receiver_engine_id:
            fields['encoder_output_receiver_engine_id'] = self.receiver_engine_id
        return fields


def build_encoder_transfer_config(receiver_endpoint_info: dict | None = None,
                                  receiver_engine_id: str | None = None,
                                  transfer_id: str | None = None) -> EncoderTransferConfig:
    """Create a validated encoder transfer config for a proxy-to-encoder request."""
    if not receiver_endpoint_info:
        raise ValueError('language node does not advertise encoder-output receiver_endpoint_info')
    return EncoderTransferConfig(
        transfer_id=transfer_id or f'epd-{uuid.uuid4().hex}',
        receiver_endpoint_info=receiver_endpoint_info,
        receiver_engine_id=receiver_engine_id,
    )


@dataclass
class _EmbeddingLayout:
    tensor: torch.Tensor
    ranges: list[list[int]]
    shapes: list[list[int]]
    dtype: str
    nbytes: int


def _jsonable(value: Any):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


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
        raise ValueError('DLSlime RDMA encoder transfer requires precomputed input_embeddings.')

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
            raise ValueError('input_embedding_ranges are required for DLSlime RDMA encoder embeddings.')
        tensor = _embedding_tensor(embedding, device)
        if tensor.ndim != 2:
            raise ValueError(f'DLSlime RDMA encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
        if end - start != tensor.shape[0]:
            raise ValueError(
                f'DLSlime RDMA encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}')
        if dtype is None:
            dtype = tensor.dtype
            hidden_size = tensor.shape[1]
        elif tensor.dtype != dtype or tensor.shape[1] != hidden_size:
            raise ValueError('DLSlime RDMA encoder embeddings must share dtype and hidden size.')
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


async def _wait_dlslime_future(future):
    wait = getattr(future, 'wait', None)
    if wait is None:
        return future
    if os.environ.get('LMDEPLOY_USE_ASYNC_MIGRATION', None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, wait)
    return wait()


class DLSlimeEncoderTransferManager:
    """Owns one process-local DLSlime endpoint for encoder-output transfer."""

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
        self.endpoint = endpoint or self._create_endpoint(link_type, ib_port, rank, device_name)
        self.encoder_cache = encoder_cache if encoder_cache is not None else get_epd_encoder_cache()
        self._connected_engine_ids: set[str] = set()
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._published_mr_keys: dict[str, str] = {}
        self._transfer_pins: dict[str, list[str]] = {}
        self._remote_mr_lock = asyncio.Lock()

    @staticmethod
    def _create_endpoint(link_type: str, ib_port: int, rank: int, device_name: str | None):
        try:
            from dlslime import RDMAEndpoint, available_nic
        except ImportError as exc:
            raise RuntimeError('dlslime is required for EPD DLSlime RDMA transfer.') from exc

        if device_name is None:
            nics = available_nic()
            if not nics:
                raise RuntimeError('no RDMA NICs are available for EPD DLSlime RDMA transfer.')
            device_name = nics[rank % len(nics)]
        logger.info(f'use device {device_name} for EPD DLSlime RDMA transfer')
        return RDMAEndpoint(device_name=device_name, ib_port=ib_port, link_type=link_type)

    @property
    def endpoint_info(self):
        return _jsonable(self.endpoint.endpoint_info())

    def _mr_info(self, transfer_id: str):
        mr_info = _jsonable(self.endpoint.mr_info())
        if isinstance(mr_info, dict) and transfer_id in mr_info:
            return mr_info[transfer_id]
        return mr_info

    def _connect_once(self, remote_engine_id: str, endpoint_info):
        if remote_engine_id in self._connected_engine_ids:
            return
        self.endpoint.connect(_jsonable(endpoint_info))
        self._connected_engine_ids.add(remote_engine_id)

    def _unregister_memory_region(self, key: str):
        unregister = getattr(self.endpoint, 'unregister_memory_region', None)
        if callable(unregister):
            unregister(key)
            return
        get_pool = getattr(self.endpoint, 'get_pool', None)
        if callable(get_pool):
            pool = get_pool()
            unregister = getattr(pool, 'unregister_memory_region', None)
            if callable(unregister):
                unregister(key)

    async def publish(self,
                      prompt_input: dict,
                      remote_engine_id: str,
                      remote_session_id: int,
                      transfer_id: str,
                      receiver_endpoint_info: dict | None = None,
                      receiver_engine_id: str | None = None) -> EncoderOutputRef:
        if receiver_endpoint_info:
            self._connect_once(receiver_engine_id or remote_engine_id, receiver_endpoint_info)
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
                raise ValueError(f'DLSlime RDMA encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
            if end - start != tensor.shape[0]:
                raise ValueError(
                    f'DLSlime RDMA encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}')
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
                    raise ValueError(f'DLSlime RDMA encoder embedding must be 2-D, got shape {tuple(tensor.shape)}')
                if end - start != tensor.shape[0]:
                    raise ValueError(
                        f'DLSlime RDMA encoder embedding range [{start}, {end}) does not match rows {tensor.shape[0]}'
                    )

                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                if entry.mr_key is None:
                    mr_key = f'epd-cache-{cache_key}'
                    self.endpoint.register_memory_region(mr_key, tensor.data_ptr(), 0, entry.nbytes)
                    entry.mr_key = mr_key

                    def _unregister_cache_entry(mr_key=entry.mr_key):
                        self._unregister_memory_region(mr_key)

                    entry.on_evict = _unregister_cache_entry

                self.encoder_cache.pin(cache_key)
                pinned_keys.append(cache_key)
                entries.append(
                    EncoderOutputEntry(
                        cache_key=cache_key,
                        mr_key=entry.mr_key,
                        mr_info=self._mr_info(entry.mr_key),
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

        self.endpoint.register_memory_region(transfer_id, layout.tensor.data_ptr(), 0, layout.nbytes)
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
                mr_info=self._mr_info(transfer_id),
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
            raise ValueError('DLSlime RDMA encoder_output_ref entry and embedding range counts do not match.')

        self._connect_once(encoder_output_ref.remote_engine_id, endpoint_info)
        embeddings = []
        for entry in entries:
            start, end = entry.input_embedding_range
            shape = entry.shape
            if len(shape) != 2:
                raise ValueError('DLSlime RDMA encoder_output_ref requires 2-D embedding shapes.')
            dtype = getattr(torch, entry.dtype)
            output = torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=self.device)
            expected_nbytes = output.numel() * output.element_size()
            nbytes = int(entry.nbytes)
            if nbytes != expected_nbytes:
                raise ValueError('DLSlime RDMA encoder_output_ref byte size does not match embedding shape and dtype.')
            if int(end) - int(start) != int(shape[0]):
                raise ValueError('DLSlime RDMA encoder_output_ref range length does not match embedding shape.')
            await self._rdma_read_encoder_output(output, entry.mr_info, nbytes)
            embeddings.append(InputEmbeddings(output, start=int(start), end=int(end)))
        return dict(prompt=None, input_ids=list(encoder_output_ref.token_ids), input_embeddings=embeddings)

    async def _receive_flat_one_shot(self, encoder_output_ref: EncoderOutputRef) -> dict:
        endpoint_info = encoder_output_ref.transfer_metadata.endpoint_info
        remote_mr_info = encoder_output_ref.transfer_metadata.mr_info
        nbytes = encoder_output_ref.transfer_metadata.nbytes
        if endpoint_info is None or remote_mr_info is None or nbytes is None:
            raise ValueError('DLSlime RDMA encoder_output_ref is missing endpoint or memory-region metadata.')

        shapes = encoder_output_ref.shape or []
        if not isinstance(shapes, list) or len(shapes) == 0 or not isinstance(shapes[0], list):
            raise ValueError('DLSlime RDMA encoder_output_ref requires per-embedding shapes.')
        ranges = encoder_output_ref.input_embedding_ranges
        if len(shapes) != len(ranges):
            raise ValueError('DLSlime RDMA encoder_output_ref shape and embedding range counts do not match.')
        dtype = getattr(torch, encoder_output_ref.dtype)
        total_rows = sum(int(shape[0]) for shape in shapes)
        hidden_size = int(shapes[0][1])
        for shape, (start, end) in zip(shapes, ranges):
            if len(shape) != 2:
                raise ValueError('DLSlime RDMA encoder_output_ref requires 2-D embedding shapes.')
            if int(shape[1]) != hidden_size:
                raise ValueError('DLSlime RDMA encoder_output_ref embeddings must share hidden size.')
            if int(end) - int(start) != int(shape[0]):
                raise ValueError('DLSlime RDMA encoder_output_ref range length does not match embedding shape.')
        output = torch.empty((total_rows, hidden_size), dtype=dtype, device=self.device)
        expected_nbytes = output.numel() * output.element_size()
        if int(nbytes) != expected_nbytes:
            raise ValueError('DLSlime RDMA encoder_output_ref byte size does not match embedding shape and dtype.')

        self._connect_once(encoder_output_ref.remote_engine_id, endpoint_info)
        await self._rdma_read_encoder_output(output, remote_mr_info, int(nbytes))

        embeddings = []
        offset = 0
        for shape, (start, end) in zip(shapes, ranges):
            rows = int(shape[0])
            embeddings.append(InputEmbeddings(output[offset:offset + rows], start=int(start), end=int(end)))
            offset += rows
        return dict(prompt=None, input_ids=list(encoder_output_ref.token_ids), input_embeddings=embeddings)

    async def _rdma_read_encoder_output(self, output: torch.Tensor, remote_mr_info, nbytes: int):
        local_key = 'epd_encoder_output_recv'
        remote_key = 'epd_encoder_output_remote'
        # Reuse fixed DLSlime MR keys to avoid accumulating remote registrations; serialize
        # the rebind/read critical section because concurrent requests share those keys.
        async with self._remote_mr_lock:
            local_handle = self.endpoint.register_memory_region(local_key, output.data_ptr(), 0, nbytes)
            remote_handle = self.endpoint.register_remote_memory_region(remote_key, _jsonable(remote_mr_info))
            try:
                future = self.endpoint.read([(local_handle, remote_handle, 0, 0, nbytes)])
                await _wait_dlslime_future(future)
            finally:
                self._unregister_memory_region(local_key)

    def release_published(self, transfer_id: str):
        for cache_key in self._transfer_pins.pop(transfer_id, []):
            if self.encoder_cache is not None:
                self.encoder_cache.unpin(cache_key)
        key = self._published_mr_keys.pop(transfer_id, None)
        if key is not None:
            self._unregister_memory_region(key)
        self._published_tensors.pop(transfer_id, None)

    def close(self):
        for transfer_id in list(self._published_tensors):
            self.release_published(transfer_id)
        if self.encoder_cache is not None:
            self.encoder_cache.clear()
        shutdown = getattr(self.endpoint, 'shutdown', None)
        if callable(shutdown):
            shutdown()


_DLSLIME_ENCODER_TRANSFER_MANAGER: DLSlimeEncoderTransferManager | None = None


def set_dlslime_encoder_transfer_manager(manager: DLSlimeEncoderTransferManager | None):
    global _DLSLIME_ENCODER_TRANSFER_MANAGER
    _DLSLIME_ENCODER_TRANSFER_MANAGER = manager


def get_dlslime_encoder_transfer_manager() -> DLSlimeEncoderTransferManager:
    if _DLSLIME_ENCODER_TRANSFER_MANAGER is None:
        raise ValueError('dlslime EPD transfer manager is not initialized.')
    return _DLSLIME_ENCODER_TRANSFER_MANAGER


async def publish_encoder_output(prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                                 transfer_config: EncoderTransferConfig) -> EncoderOutputRef:
    """Publish computed encoder output through DLSlime."""
    manager = get_dlslime_encoder_transfer_manager()
    return await manager.publish(prompt_input,
                                 remote_engine_id,
                                 remote_session_id,
                                 transfer_config.transfer_id,
                                 receiver_endpoint_info=transfer_config.receiver_endpoint_info,
                                 receiver_engine_id=transfer_config.receiver_engine_id)


async def load_encoder_output_async(encoder_output_ref: EncoderOutputRef) -> dict:
    """Convert an encoder cache ref into prompt input through DLSlime."""
    manager = get_dlslime_encoder_transfer_manager()
    try:
        return await manager.receive(encoder_output_ref)
    finally:
        asyncio.create_task(free_remote_encoder_cache_ref_async(encoder_output_ref))


async def free_published_encoder_cache_ref_async(request: EncoderCacheFreeRequest) -> None:
    """Free producer-side encoder cache reference state."""
    manager = get_dlslime_encoder_transfer_manager()
    manager.release_published(request.transfer_id)


async def free_remote_encoder_cache_ref_async(encoder_output_ref: EncoderOutputRef) -> None:
    """Free remote producer state after a DLSlime transfer is consumed."""
    if not encoder_output_ref.transfer_id:
        return
    if not encoder_output_ref.remote_engine_id.startswith(('http://', 'https://')):
        logger.debug('skip EPD encoder cache ref free for non-http engine id %s', encoder_output_ref.remote_engine_id)
        return

    request = EncoderCacheFreeRequest(transfer_id=encoder_output_ref.transfer_id)
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(encoder_output_ref.remote_engine_id.rstrip('/') + '/epd/free_encoder_cache_ref',
                                    json=request.model_dump(mode='json')) as response:
                if response.status != 200:
                    logger.warning('EPD encoder cache ref free failed: status=%s, body=%s', response.status,
                                   await response.text())
    except Exception as exc:
        logger.warning('EPD encoder cache ref free failed: %s', exc)
