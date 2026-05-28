# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator

from lmdeploy.pytorch.disagg.config import (
    DistServeEngineConfig,
    DistServeNVLinkConfig,
    DistServeRDMAConfig,
    DistServeTCPConfig,
)


class MigrationProtocol(enum.Enum):
    """Migration Transport Protocol.

    Attributes:
        RDMA: IB or RoCEv1/v2.
        NVLINK: High device-to-device link.

    Warning: By now, only `GPU Directed RDMA` is supported in DistServe.
        We preserve several protocol and will be implemented in the future.
    """

    TCP = enum.auto()
    RDMA = enum.auto()
    NVLINK = enum.auto()


class DistServeConnectionStatus(enum.Enum):
    # TODO(JimyMa): Add more connection failure handler
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class DistServeInitRequest(BaseModel):
    local_engine_id: str
    local_engine_config: DistServeEngineConfig

    remote_engine_id: str
    remote_engine_config: DistServeEngineConfig

    protocol: MigrationProtocol

    rank: int | None = None

    tcp_config: DistServeTCPConfig | None = None
    rdma_config: DistServeRDMAConfig | None = None
    nvlink_config: DistServeNVLinkConfig | None = None


class DistServeEngineEndpointInfo(BaseModel):
    zmq_address: str


class DistServeKVTransferEndpointInfo(BaseModel):
    protocol: MigrationProtocol
    endpoint_info: str


class DistServeInitResponse(BaseModel):
    status: DistServeConnectionStatus
    # the control plane initialization feedback
    engine_endpoint_info: DistServeEngineEndpointInfo
    # the KVCache Transfer initialization feedback
    # To ensure generality (where endpoint_info can be initialization information
    # for different media such as RDMA, NVLink, etc.), we use a string (str) to
    # store this information.
    kvtransfer_endpoint_info: list[DistServeKVTransferEndpointInfo]


class DistServeConnectionRequest(BaseModel):
    protocol: MigrationProtocol
    remote_engine_id: str
    remote_engine_endpoint_info: DistServeEngineEndpointInfo
    remote_kvtransfer_endpoint_info: list[DistServeKVTransferEndpointInfo]


class DistServeConnectionResponse(BaseModel):
    status: DistServeConnectionStatus


class MigrationRequest(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: list[int]

    is_dummy_prefill: bool = False


class EncoderHttpJsonEmbedding(BaseModel):
    """HTTP JSON encoder embedding payload for the first EPD receive path.

    This is intentionally a small bring-up format. A production transfer
    backend should replace the HTTP JSON data with backend-specific cache
    handles.
    """

    data: list[list[float]]
    start: int
    end: int
    dtype: str | None = None


class EncoderCacheRef(BaseModel):
    """Reference to encoder outputs produced by an EPD encoder node."""

    token_ids: list[int]
    mm_mask: list[int] | None = None
    input_embedding_ranges: list[list[int]] | None = None
    input_embeddings: list[EncoderHttpJsonEmbedding] | None = None
    transfer_id: str | None = None

    protocol: MigrationProtocol
    backend: str = 'http_json'
    remote_engine_id: str
    remote_session_id: int
    remote_block_ids: list[int] = Field(default_factory=list)

    dtype: str | None = None
    shape: list[int] | list[list[int]] | None = None
    modality: str | list[str] | None = None
    cache_key: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator('protocol', mode='before')
    @classmethod
    def _validate_protocol(cls, value):
        if isinstance(value, str):
            return MigrationProtocol[value]
        return value

    @field_serializer('protocol')
    def _serialize_protocol(self, protocol: MigrationProtocol):
        return protocol.name


class DistServeCacheFreeRequest(BaseModel):
    remote_engine_id: str
    remote_session_id: int


class DistServeDropConnectionRequest(BaseModel):
    engine_id: str
    remote_engine_id: str
