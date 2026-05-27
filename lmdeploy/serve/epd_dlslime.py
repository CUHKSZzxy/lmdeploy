# Copyright (c) OpenMMLab. All rights reserved.
"""DLSlime/RDMA manager for EPD encoder-output transfer."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, MigrationProtocol
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.serve.epd_channel import EPD_BACKEND_DLSLIME_RDMA
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_DLSLIME_EXTRA_ENDPOINT_INFO = 'dlslime_endpoint_info'
_DLSLIME_EXTRA_MR_INFO = 'dlslime_mr_info'
_DLSLIME_EXTRA_NBYTES = 'dlslime_nbytes'

_TORCH_DTYPES = {
    'float32': torch.float32,
    'torch.float32': torch.float32,
    'float16': torch.float16,
    'torch.float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'torch.bfloat16': torch.bfloat16,
}


@dataclass
class _EmbeddingLayout:
    tensor: torch.Tensor
    ranges: list[list[int]]
    shapes: list[list[int]]
    dtype: str
    nbytes: int


def _to_int_list(values) -> list[int]:
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return [int(value) for value in values]


def _embedding_range(embedding, ranges, index: int) -> list[int]:
    if ranges is not None:
        return _to_int_list(ranges[index])
    if hasattr(embedding, 'start') and hasattr(embedding, 'end'):
        return [int(embedding.start), int(embedding.end)]
    raise ValueError('input_embedding_ranges are required for DLSlime RDMA encoder embeddings.')


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    return name.split('.', 1)[1] if name.startswith('torch.') else name


def _torch_dtype(dtype: str | None) -> torch.dtype:
    if dtype not in _TORCH_DTYPES:
        raise ValueError(f'unsupported DLSlime RDMA encoder embedding dtype: {dtype}')
    return _TORCH_DTYPES[dtype]


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
        start, end = _embedding_range(embedding, input_embedding_ranges, index)
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
        dtype=_dtype_name(flat_tensor.dtype),
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


class DlslimeRdmaTransferManager:
    """Owns one process-local DLSlime endpoint for encoder-output transfer."""

    def __init__(self,
                 engine_id: str,
                 *,
                 device: str | torch.device | None = None,
                 endpoint=None,
                 link_type: str = 'RoCE',
                 ib_port: int = 1,
                 rank: int = 0,
                 device_name: str | None = None):
        self.engine_id = engine_id
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.endpoint = endpoint or self._create_endpoint(link_type, ib_port, rank, device_name)
        self._connected_engine_ids: set[str] = set()
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._received_tensors: dict[str, torch.Tensor] = {}

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

    async def publish(self,
                      prompt_input: dict,
                      remote_engine_id: str,
                      remote_session_id: int,
                      transfer_id: str,
                      receiver_endpoint_info: dict | None = None,
                      receiver_engine_id: str | None = None) -> EncoderCacheRef:
        if receiver_endpoint_info:
            self._connect_once(receiver_engine_id or remote_engine_id, receiver_endpoint_info)
        layout = _build_embedding_layout(prompt_input, self.device)
        self.endpoint.register_memory_region(transfer_id, layout.tensor.data_ptr(), 0, layout.nbytes)
        self._published_tensors[transfer_id] = layout.tensor

        return EncoderCacheRef(
            token_ids=_to_int_list(prompt_input.get('input_ids') or []),
            input_embedding_ranges=layout.ranges,
            protocol=MigrationProtocol.RDMA,
            backend=EPD_BACKEND_DLSLIME_RDMA,
            transfer_id=transfer_id,
            remote_engine_id=remote_engine_id,
            remote_session_id=remote_session_id,
            remote_block_ids=[],
            dtype=layout.dtype,
            shape=layout.shapes,
            extra={
                _DLSLIME_EXTRA_ENDPOINT_INFO: self.endpoint_info,
                _DLSLIME_EXTRA_MR_INFO: self._mr_info(transfer_id),
                _DLSLIME_EXTRA_NBYTES: layout.nbytes,
            },
        )

    async def receive(self, encoder_result: EncoderCacheRef) -> dict:
        if not encoder_result.transfer_id:
            raise ValueError('DLSlime RDMA encoder_result requires transfer_id.')
        endpoint_info = encoder_result.extra.get(_DLSLIME_EXTRA_ENDPOINT_INFO)
        remote_mr_info = encoder_result.extra.get(_DLSLIME_EXTRA_MR_INFO)
        nbytes = encoder_result.extra.get(_DLSLIME_EXTRA_NBYTES)
        if endpoint_info is None or remote_mr_info is None or nbytes is None:
            raise ValueError('DLSlime RDMA encoder_result is missing endpoint or memory-region metadata.')

        shapes = encoder_result.shape or []
        if not isinstance(shapes, list) or len(shapes) == 0 or not isinstance(shapes[0], list):
            raise ValueError('DLSlime RDMA encoder_result requires per-embedding shapes.')
        dtype = _torch_dtype(encoder_result.dtype)
        total_rows = sum(int(shape[0]) for shape in shapes)
        hidden_size = int(shapes[0][1])
        output = torch.empty((total_rows, hidden_size), dtype=dtype, device=self.device)

        local_key = f'{encoder_result.transfer_id}:recv'
        remote_key = f'{encoder_result.transfer_id}:remote'
        local_handle = self.endpoint.register_memory_region(local_key, output.data_ptr(), 0, int(nbytes))
        self._connect_once(encoder_result.remote_engine_id, endpoint_info)
        remote_handle = self.endpoint.register_remote_memory_region(remote_key, _jsonable(remote_mr_info))
        future = self.endpoint.read([(local_handle, remote_handle, 0, 0, int(nbytes))])
        await _wait_dlslime_future(future)
        self._received_tensors[encoder_result.transfer_id] = output

        ranges = encoder_result.input_embedding_ranges or []
        embeddings = []
        offset = 0
        for shape, (start, end) in zip(shapes, ranges):
            rows = int(shape[0])
            embeddings.append(InputEmbeddings(output[offset:offset + rows], start=int(start), end=int(end)))
            offset += rows
        return dict(prompt=None, input_ids=list(encoder_result.token_ids), input_embeddings=embeddings)

    def close(self):
        shutdown = getattr(self.endpoint, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        self._published_tensors.clear()
        self._received_tensors.clear()


_DLSLIME_RDMA_MANAGER: DlslimeRdmaTransferManager | None = None


def set_dlslime_rdma_transfer_manager(manager: DlslimeRdmaTransferManager | None):
    global _DLSLIME_RDMA_MANAGER
    _DLSLIME_RDMA_MANAGER = manager


def get_dlslime_rdma_transfer_manager() -> DlslimeRdmaTransferManager:
    if _DLSLIME_RDMA_MANAGER is None:
        raise ValueError('dlslime_rdma EPD transfer manager is not initialized.')
    return _DLSLIME_RDMA_MANAGER
