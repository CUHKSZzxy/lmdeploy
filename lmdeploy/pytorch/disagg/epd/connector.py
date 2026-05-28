# Copyright (c) OpenMMLab. All rights reserved.
"""Connector boundary for EPD encoder-output transfer."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, EncoderHttpJsonEmbedding, MigrationProtocol
from lmdeploy.pytorch.messages import InputEmbeddings

from .channel import (
    EPD_BACKEND_DLSLIME_RDMA,
    EPD_BACKEND_HTTP_JSON,
    EPD_TRANSFER_BACKENDS,
    EncoderTransferEmbedding,
)
from .dlslime import get_dlslime_rdma_transfer_manager


@dataclass(frozen=True)
class EncoderTransferConfig:
    """Per-request encoder transfer configuration."""

    backend: str = EPD_BACKEND_HTTP_JSON
    transfer_id: str | None = None
    receiver_endpoint_info: dict | None = None
    receiver_engine_id: str | None = None

    def __post_init__(self):
        if self.backend not in EPD_TRANSFER_BACKENDS:
            raise ValueError(f'unsupported EPD encoder transfer backend: {self.backend}')
        if self.backend == EPD_BACKEND_DLSLIME_RDMA:
            if not self.transfer_id:
                raise ValueError('EPD DLSlime RDMA transfer requires transfer_id')
            if not self.receiver_endpoint_info:
                raise ValueError('EPD DLSlime RDMA transfer requires receiver_endpoint_info')

    @classmethod
    def from_request(cls, request_dict: dict, default_backend: str = EPD_BACKEND_HTTP_JSON) -> 'EncoderTransferConfig':
        """Build transfer config from internal request fields."""
        return cls(
            backend=request_dict.get('encoder_transfer_backend') or default_backend or EPD_BACKEND_HTTP_JSON,
            transfer_id=request_dict.get('epd_transfer_id'),
            receiver_endpoint_info=request_dict.get('encoder_output_receiver_endpoint_info'),
            receiver_engine_id=request_dict.get('encoder_output_receiver_engine_id'),
        )

    def to_request_fields(self) -> dict:
        """Serialize transfer config into internal request fields."""
        fields = {'encoder_transfer_backend': self.backend}
        if self.transfer_id:
            fields['epd_transfer_id'] = self.transfer_id
        if self.receiver_endpoint_info:
            fields['encoder_output_receiver_endpoint_info'] = self.receiver_endpoint_info
        if self.receiver_engine_id:
            fields['encoder_output_receiver_engine_id'] = self.receiver_engine_id
        return fields


def build_encoder_transfer_config(backend: str | None,
                                  receiver_endpoint_info: dict | None = None,
                                  receiver_engine_id: str | None = None,
                                  transfer_id: str | None = None) -> EncoderTransferConfig:
    """Create a validated encoder transfer config for a proxy-to-encoder request."""
    backend = backend or EPD_BACKEND_HTTP_JSON
    if backend not in EPD_TRANSFER_BACKENDS:
        raise ValueError(f'unsupported EPD encoder transfer backend: {backend}')
    if backend == EPD_BACKEND_DLSLIME_RDMA:
        if not receiver_endpoint_info:
            raise ValueError('language node does not advertise encoder-output receiver_endpoint_info')
        transfer_id = transfer_id or f'epd-{uuid.uuid4().hex}'
        return EncoderTransferConfig(backend=backend,
                                     transfer_id=transfer_id,
                                     receiver_endpoint_info=receiver_endpoint_info,
                                     receiver_engine_id=receiver_engine_id)
    return EncoderTransferConfig(backend=backend)


class EncoderTransferConnector(ABC):
    """Backend boundary for publishing and receiving encoder outputs."""

    backend: str

    @abstractmethod
    async def publish(self, prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                      transfer_config: EncoderTransferConfig) -> EncoderCacheRef:
        """Publish encoder outputs and return a serializable reference."""
        raise NotImplementedError

    @abstractmethod
    async def receive(self, encoder_result: EncoderCacheRef) -> dict:
        """Receive encoder outputs and return language-engine prompt input."""
        raise NotImplementedError


class HttpJsonEncoderTransferConnector(EncoderTransferConnector):
    """Transfer encoder embeddings directly in the HTTP JSON response."""

    backend = EPD_BACKEND_HTTP_JSON

    async def publish(self, prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                      transfer_config: EncoderTransferConfig) -> EncoderCacheRef:
        return prompt_input_to_encoder_cache_ref(
            prompt_input,
            remote_engine_id=remote_engine_id,
            remote_session_id=remote_session_id,
            backend=self.backend,
        )

    async def receive(self, encoder_result: EncoderCacheRef) -> dict:
        return encoder_cache_ref_to_prompt_input(encoder_result)


class DlslimeRdmaEncoderTransferConnector(EncoderTransferConnector):
    """Transfer encoder embeddings through DLSlime RDMA."""

    backend = EPD_BACKEND_DLSLIME_RDMA

    async def publish(self, prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                      transfer_config: EncoderTransferConfig) -> EncoderCacheRef:
        manager = get_dlslime_rdma_transfer_manager()
        return await manager.publish(prompt_input,
                                     remote_engine_id,
                                     remote_session_id,
                                     transfer_config.transfer_id,
                                     receiver_endpoint_info=transfer_config.receiver_endpoint_info,
                                     receiver_engine_id=transfer_config.receiver_engine_id)

    async def receive(self, encoder_result: EncoderCacheRef) -> dict:
        manager = get_dlslime_rdma_transfer_manager()
        return await manager.receive(encoder_result)


_CONNECTORS: dict[str, EncoderTransferConnector] = {
    EPD_BACKEND_HTTP_JSON: HttpJsonEncoderTransferConnector(),
    EPD_BACKEND_DLSLIME_RDMA: DlslimeRdmaEncoderTransferConnector(),
}


def get_encoder_transfer_connector(backend: str) -> EncoderTransferConnector:
    """Return the connector implementation for a backend."""
    connector = _CONNECTORS.get(backend)
    if connector is None:
        raise ValueError(f'unsupported EPD encoder transfer backend: {backend}')
    return connector


async def publish_encoder_output(prompt_input: dict, remote_engine_id: str, remote_session_id: int,
                                 transfer_config: EncoderTransferConfig) -> EncoderCacheRef:
    """Publish computed encoder output through the configured backend."""
    connector = get_encoder_transfer_connector(transfer_config.backend)
    return await connector.publish(prompt_input, remote_engine_id, remote_session_id, transfer_config)


async def load_encoder_output_async(encoder_result: EncoderCacheRef) -> dict:
    """Convert an encoder cache ref into prompt input through its backend connector."""
    connector = get_encoder_transfer_connector(encoder_result.backend)
    return await connector.receive(encoder_result)


_NUMPY_DTYPES = {
    None: np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': np.float32,
}


def _as_numpy_dtype(dtype: str | None):
    """Map serialized embedding dtype names to numpy dtypes."""
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f'unsupported HTTP JSON encoder embedding dtype: {dtype}')
    return _NUMPY_DTYPES[dtype]


def _to_int_list(values) -> list[int]:
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return [int(value) for value in values]


def _embedding_array_and_dtype(embedding) -> tuple[np.ndarray, str]:
    """Convert one embedding payload to a numpy array and serialize its dtype."""
    data = getattr(embedding, 'embeddings', embedding)
    dtype = str(getattr(data, 'dtype', 'float32'))
    if dtype.startswith('torch.'):
        dtype = dtype.split('.', 1)[1]
    if hasattr(data, 'detach'):
        if dtype == 'bfloat16':
            data = data.float()
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)
        dtype = str(data.dtype)
    return np.asarray(data), dtype


def _embedding_range(embedding, ranges, index: int) -> list[int]:
    if ranges is not None:
        return _to_int_list(ranges[index])
    if hasattr(embedding, 'start') and hasattr(embedding, 'end'):
        return [int(embedding.start), int(embedding.end)]
    raise ValueError('input_embedding_ranges are required to serialize encoder embeddings.')


def _embedding_payloads(prompt_input: dict) -> tuple[list[EncoderTransferEmbedding], list[list[int]], list[list[int]],
                                                    list[str]]:
    input_embeddings = prompt_input.get('input_embeddings') or []
    input_embedding_ranges = prompt_input.get('input_embedding_ranges')
    payload_embeddings: list[EncoderTransferEmbedding] = []
    shapes: list[list[int]] = []
    dtypes: list[str] = []

    for index, embedding in enumerate(input_embeddings):
        start, end = _embedding_range(embedding, input_embedding_ranges, index)
        data, _ = _embedding_array_and_dtype(embedding)
        if data.ndim != 2:
            raise ValueError(f'encoder embedding must be 2-D, got shape {data.shape}')
        if end - start != data.shape[0]:
            raise ValueError(f'encoder embedding range [{start}, {end}) does not match embedding rows {data.shape[0]}')
        data = np.ascontiguousarray(data)
        dtype = str(data.dtype)
        payload_embeddings.append(EncoderTransferEmbedding(data=data, start=start, end=end, dtype=dtype))
        shapes.append(list(data.shape))
        dtypes.append(dtype)

    ranges = [[emb.start, emb.end] for emb in payload_embeddings]
    return payload_embeddings, ranges, shapes, dtypes


def encoder_cache_ref_to_prompt_input(encoder_result: EncoderCacheRef) -> dict:
    """Convert a HTTP JSON encoder cache reference into prompt input."""
    if encoder_result.backend != EPD_BACKEND_HTTP_JSON:
        raise ValueError(f'EPD backend {encoder_result.backend!r} requires async connector receive.')

    prompt_input = dict(prompt=None, input_ids=list(encoder_result.token_ids))
    if not encoder_result.input_embeddings:
        return prompt_input

    embeddings = []
    for http_json_embedding in encoder_result.input_embeddings:
        dtype = _as_numpy_dtype(http_json_embedding.dtype or encoder_result.dtype)
        data = np.asarray(http_json_embedding.data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError(f'HTTP JSON encoder embedding must be 2-D, got shape {data.shape}')
        expected_rows = http_json_embedding.end - http_json_embedding.start
        if expected_rows != data.shape[0]:
            raise ValueError(
                f'HTTP JSON encoder embedding range [{http_json_embedding.start}, {http_json_embedding.end}) '
                f'does not match embedding rows {data.shape[0]}')
        embeddings.append(InputEmbeddings(data, start=http_json_embedding.start, end=http_json_embedding.end))

    prompt_input['input_embeddings'] = embeddings
    return prompt_input


def prompt_input_to_encoder_cache_ref(prompt_input: dict,
                                      remote_engine_id: str,
                                      remote_session_id: int,
                                      protocol: MigrationProtocol = MigrationProtocol.TCP,
                                      backend: str = EPD_BACKEND_HTTP_JSON,
                                      transfer_id: str | None = None) -> EncoderCacheRef:
    """Serialize prompt-side embeddings into an EPD encoder cache ref."""
    if backend not in EPD_TRANSFER_BACKENDS:
        raise ValueError(f'unsupported EPD encoder transfer backend: {backend}')
    input_ids = _to_int_list(prompt_input.get('input_ids') or [])
    input_embeddings = prompt_input.get('input_embeddings')
    if not input_embeddings:
        if prompt_input.get('multimodal') or prompt_input.get('input_multimodals'):
            raise ValueError('EPD encoder producer requires precomputed input_embeddings.')
        return EncoderCacheRef(token_ids=input_ids,
                               protocol=protocol,
                               backend=backend,
                               transfer_id=transfer_id,
                               remote_engine_id=remote_engine_id,
                               remote_session_id=remote_session_id,
                               remote_block_ids=[])

    payload_embeddings, ranges, shapes, dtypes = _embedding_payloads(prompt_input)
    http_json_embeddings = None
    if backend == EPD_BACKEND_HTTP_JSON:
        http_json_embeddings = [
            EncoderHttpJsonEmbedding(data=embedding.data.astype(np.float32).tolist(),
                                     start=embedding.start,
                                     end=embedding.end,
                                     dtype=embedding.dtype) for embedding in payload_embeddings
        ]
    elif not transfer_id:
        raise ValueError('non-http_json EPD encoder_result requires transfer_id')

    return EncoderCacheRef(token_ids=input_ids,
                           input_embedding_ranges=ranges,
                           input_embeddings=http_json_embeddings,
                           protocol=protocol,
                           backend=backend,
                           transfer_id=transfer_id,
                           remote_engine_id=remote_engine_id,
                           remote_session_id=remote_session_id,
                           remote_block_ids=[],
                           dtype=dtypes[0] if len(set(dtypes)) == 1 else None,
                           shape=shapes)
