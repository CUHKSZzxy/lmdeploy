# Copyright (c) OpenMMLab. All rights reserved.
"""DLSlime data-plane helpers for EPD encoder-output transfer."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _jsonable(value: Any):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


async def _wait_dlslime_future(future):
    wait = getattr(future, 'wait', None)
    if wait is None:
        return future
    if os.environ.get('LMDEPLOY_USE_ASYNC_MIGRATION', None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, wait)
    return wait()


class DLSlimeEndpoint:
    """Small wrapper around one DLSlime RDMA endpoint."""

    def __init__(self,
                 *,
                 endpoint=None,
                 link_type: str = 'RoCE',
                 ib_port: int = 1,
                 rank: int = 0,
                 device_name: str | None = None):
        self.endpoint = endpoint or self._create_endpoint(link_type, ib_port, rank, device_name)
        self._connected_engine_ids: set[str] = set()
        self._remote_mr_lock = asyncio.Lock()

    @staticmethod
    def _create_endpoint(link_type: str, ib_port: int, rank: int, device_name: str | None):
        try:
            from dlslime import RDMAEndpoint, available_nic
        except ImportError as exc:
            raise RuntimeError('dlslime is required for EPD encoder transfer.') from exc

        if device_name is None:
            nics = available_nic()
            if not nics:
                raise RuntimeError('no RDMA NICs are available for EPD encoder transfer.')
            device_name = nics[rank % len(nics)]
        logger.info(f'use device {device_name} for EPD encoder transfer')
        return RDMAEndpoint(device_name=device_name, ib_port=ib_port, link_type=link_type)

    @property
    def endpoint_info(self):
        return _jsonable(self.endpoint.endpoint_info())

    def mr_info(self, key: str):
        mr_info = _jsonable(self.endpoint.mr_info())
        if isinstance(mr_info, dict) and key in mr_info:
            return mr_info[key]
        return mr_info

    def connect_once(self, remote_engine_id: str, endpoint_info):
        if remote_engine_id in self._connected_engine_ids:
            return
        self.endpoint.connect(_jsonable(endpoint_info))
        self._connected_engine_ids.add(remote_engine_id)

    def drop_connect(self, remote_engine_id: str):
        self._connected_engine_ids.discard(remote_engine_id)

    def register_memory_region(self, key: str, data_ptr: int, nbytes: int):
        return self.endpoint.register_memory_region(key, data_ptr, 0, nbytes)

    def unregister_memory_region(self, key: str):
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

    async def read_into(self, output: torch.Tensor, remote_mr_info, nbytes: int):
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
                self.unregister_memory_region(local_key)

    def close(self):
        shutdown = getattr(self.endpoint, 'shutdown', None)
        if callable(shutdown):
            shutdown()
