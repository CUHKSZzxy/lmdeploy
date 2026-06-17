# Copyright (c) OpenMMLab. All rights reserved.
"""Control-plane helpers for EPD encoder-output transfer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import aiohttp

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheFreeRequest, EncoderOutputRef
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass(frozen=True)
class EncoderTransferConfig:
    """Per-request encoder transfer configuration."""

    transfer_id: str | None = None
    receiver_engine_id: str | None = None

    def __post_init__(self):
        if not self.transfer_id:
            raise ValueError('EPD encoder transfer requires transfer_id')
        if not self.receiver_engine_id:
            raise ValueError('EPD encoder transfer requires receiver_engine_id')

    @classmethod
    def from_request(cls, request_dict: dict) -> 'EncoderTransferConfig':
        """Build transfer config from internal request fields."""
        return cls(
            transfer_id=request_dict.get('epd_transfer_id'),
            receiver_engine_id=request_dict.get('encoder_output_receiver_engine_id'),
        )

    def to_request_fields(self) -> dict:
        """Serialize transfer config into internal request fields."""
        return {
            'epd_transfer_id': self.transfer_id,
            'encoder_output_receiver_engine_id': self.receiver_engine_id,
        }


def build_encoder_transfer_config(receiver_engine_id: str | None = None,
                                  transfer_id: str | None = None) -> EncoderTransferConfig:
    """Create a validated encoder transfer config for a proxy-to-encoder request."""
    return EncoderTransferConfig(
        transfer_id=transfer_id or f'epd-{uuid.uuid4().hex}',
        receiver_engine_id=receiver_engine_id,
    )


async def free_remote_encoder_cache_ref_async(encoder_output_ref: EncoderOutputRef) -> None:
    """Free remote producer state after an encoder-output transfer is consumed."""
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
