# Copyright (c) OpenMMLab. All rights reserved.
"""EPD encoder embedding transfer channels."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import zmq
import zmq.asyncio

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

EPD_BACKEND_HTTP_JSON = 'http_json'
EPD_BACKEND_ZMQ_IPC = 'zmq_ipc'
EPD_TRANSFER_BACKENDS = (EPD_BACKEND_HTTP_JSON, EPD_BACKEND_ZMQ_IPC)


@dataclass
class EncoderTransferEmbedding:
    data: np.ndarray
    start: int
    end: int
    dtype: str


@dataclass
class EncoderTransferPayload:
    transfer_id: str
    token_ids: list[int]
    embeddings: list[EncoderTransferEmbedding]
    modality: str | list[str] | None = None
    cache_key: str | None = None
    extra: dict | None = None

    @property
    def input_embedding_ranges(self) -> list[list[int]]:
        return [[embedding.start, embedding.end] for embedding in self.embeddings]


def default_epd_channel_address(server_port: int) -> str:
    """Build the default local IPC endpoint for an EPD language node."""
    return f'ipc:///tmp/lmdeploy_epd_{server_port}.sock'


def _ipc_path(address: str) -> str | None:
    if not address.startswith('ipc://'):
        return None
    return address[len('ipc://'):]


def _payload_to_frames(payload: EncoderTransferPayload) -> list[bytes]:
    frames = []
    embedding_headers = []
    for embedding in payload.embeddings:
        data = np.ascontiguousarray(embedding.data)
        embedding_headers.append(
            dict(
                dtype=str(data.dtype),
                shape=list(data.shape),
                start=int(embedding.start),
                end=int(embedding.end),
            ))
        frames.append(data.tobytes(order='C'))

    header = dict(
        transfer_id=payload.transfer_id,
        token_ids=list(payload.token_ids),
        input_embedding_ranges=payload.input_embedding_ranges,
        embeddings=embedding_headers,
        modality=payload.modality,
        cache_key=payload.cache_key,
        extra=payload.extra or {},
    )
    return [json.dumps(header).encode()] + frames


def _frames_to_payload(frames: list[bytes]) -> EncoderTransferPayload:
    if not frames:
        raise ValueError('empty EPD channel payload')
    header = json.loads(frames[0].decode())
    embedding_headers = header.get('embeddings') or []
    if len(embedding_headers) != len(frames) - 1:
        raise ValueError('EPD channel embedding frame count mismatch')

    embeddings = []
    for emb_header, frame in zip(embedding_headers, frames[1:]):
        dtype = np.dtype(emb_header['dtype'])
        shape = tuple(int(dim) for dim in emb_header['shape'])
        data = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
        embeddings.append(
            EncoderTransferEmbedding(
                data=data,
                start=int(emb_header['start']),
                end=int(emb_header['end']),
                dtype=str(dtype),
            ))

    return EncoderTransferPayload(
        transfer_id=header['transfer_id'],
        token_ids=[int(token_id) for token_id in header.get('token_ids', [])],
        embeddings=embeddings,
        modality=header.get('modality'),
        cache_key=header.get('cache_key'),
        extra=header.get('extra') or {},
    )


class ZmqIpcEncoderSender:
    """Async ZMQ sender for EPD embedding payloads."""

    def __init__(self, address: str):
        self.address = address
        self.context = zmq.asyncio.Context(1)
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(address)
        self._lock = asyncio.Lock()

    async def send_async(self, payload: EncoderTransferPayload):
        frames = _payload_to_frames(payload)
        async with self._lock:
            await self.socket.send_multipart(frames)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqIpcEncoderReceiver:
    """Async ZMQ receiver with a small transfer-id keyed payload buffer."""

    def __init__(self, address: str, timeout: float = 30.0, max_entries: int = 1024):
        self.address = address
        self.timeout = timeout
        self.max_entries = max_entries
        self.context = zmq.asyncio.Context(1)
        self.socket = self.context.socket(zmq.PULL)
        self._buffer: dict[str, tuple[float, EncoderTransferPayload]] = {}
        self._waiters: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start(self):
        path = _ipc_path(self.address)
        if path and os.path.exists(path):
            os.unlink(path)
        self.socket.bind(self.address)
        self.address = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._task = asyncio.create_task(self._recv_loop())
        logger.info(f'EPD ZMQ IPC receiver started at {self.address}')

    async def _recv_loop(self):
        while True:
            frames = await self.socket.recv_multipart()
            payload = _frames_to_payload(frames)
            await self._put_payload(payload)

    async def _put_payload(self, payload: EncoderTransferPayload):
        async with self._lock:
            self._cleanup_locked()
            waiter = self._waiters.pop(payload.transfer_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(payload)
                return
            if len(self._buffer) >= self.max_entries:
                oldest = min(self._buffer.items(), key=lambda item: item[1][0])[0]
                self._buffer.pop(oldest, None)
            self._buffer[payload.transfer_id] = (time.monotonic(), payload)

    def _cleanup_locked(self):
        now = time.monotonic()
        expired = [key for key, (created, _) in self._buffer.items() if now - created > self.timeout]
        for key in expired:
            self._buffer.pop(key, None)

    async def recv_async(self, transfer_id: str, timeout: float | None = None) -> EncoderTransferPayload:
        async with self._lock:
            self._cleanup_locked()
            buffered = self._buffer.pop(transfer_id, None)
            if buffered is not None:
                return buffered[1]
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._waiters[transfer_id] = future

        try:
            return await asyncio.wait_for(future, timeout or self.timeout)
        finally:
            async with self._lock:
                if self._waiters.get(transfer_id) is future:
                    self._waiters.pop(transfer_id, None)

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.socket.close(linger=0)
        self.context.term()
        path = _ipc_path(self.address)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


class _SenderPool:

    def __init__(self):
        self._senders: dict[str, ZmqIpcEncoderSender] = {}

    def get(self, address: str) -> ZmqIpcEncoderSender:
        sender = self._senders.get(address)
        if sender is None:
            sender = ZmqIpcEncoderSender(address)
            self._senders[address] = sender
        return sender

    def close_all(self):
        for sender in self._senders.values():
            sender.close()
        self._senders.clear()


_sender_pool = _SenderPool()
_receiver: ZmqIpcEncoderReceiver | None = None


async def send_epd_payload(address: str, payload: EncoderTransferPayload):
    """Send an encoder payload to a language-node EPD receiver."""
    await _sender_pool.get(address).send_async(payload)


async def start_epd_receiver(address: str, timeout: float = 30.0) -> ZmqIpcEncoderReceiver:
    """Start the process-global EPD receiver."""
    global _receiver
    if _receiver is not None:
        return _receiver
    _receiver = ZmqIpcEncoderReceiver(address, timeout=timeout)
    try:
        await _receiver.start()
    except Exception:
        _receiver = None
        raise
    return _receiver


async def stop_epd_receiver():
    """Stop the process-global EPD receiver."""
    global _receiver
    if _receiver is not None:
        await _receiver.close()
        _receiver = None


async def recv_epd_payload(transfer_id: str, timeout: float | None = None) -> EncoderTransferPayload:
    """Receive an encoder payload by transfer id."""
    if _receiver is None:
        raise RuntimeError('EPD receiver is not started')
    return await _receiver.recv_async(transfer_id, timeout=timeout)


def close_epd_senders():
    """Close process-global EPD sender sockets."""
    _sender_pool.close_all()
