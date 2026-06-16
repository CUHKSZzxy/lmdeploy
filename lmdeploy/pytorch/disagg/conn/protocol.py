# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Any

from pydantic import BaseModel, field_serializer, field_validator

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


class EncoderOutputEntry(BaseModel):
    """Metadata for one cached encoder-output tensor."""

    mr_info: dict[str, Any]
    shape: list[int]
    dtype: str
    nbytes: int
    input_embedding_range: list[int]
    cache_key: str | None = None
    mr_key: str | None = None


class EncoderOutputMetadata(BaseModel):
    """Metadata needed to load encoder outputs from a producer."""

    endpoint_info: dict[str, Any]
    mr_info: dict[str, Any] | None = None
    nbytes: int | None = None
    entries: list[EncoderOutputEntry] | None = None


class EncoderOutputRef(BaseModel):
    """Reference to encoder outputs produced by an EPD encoder node."""

    token_ids: list[int]
    input_embedding_ranges: list[list[int]]
    transfer_id: str

    protocol: MigrationProtocol
    remote_engine_id: str
    remote_session_id: int

    dtype: str
    shape: list[list[int]]
    transfer_metadata: EncoderOutputMetadata

    @field_validator('protocol', mode='before')
    @classmethod
    def _validate_protocol(cls, value):
        if isinstance(value, str):
            return MigrationProtocol[value]
        return value

    @field_serializer('protocol')
    def _serialize_protocol(self, protocol: MigrationProtocol):
        return protocol.name


class EncoderCacheFreeRequest(BaseModel):
    """Request to free process-local encoder transfer state."""
    transfer_id: str


class DistServeCacheFreeRequest(BaseModel):
    remote_engine_id: str
    remote_session_id: int


class DistServeDropConnectionRequest(BaseModel):
    engine_id: str
    remote_engine_id: str
