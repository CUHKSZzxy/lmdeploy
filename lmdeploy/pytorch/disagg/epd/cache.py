# Copyright (c) OpenMMLab. All rights reserved.
"""Producer-side GPU cache for EPD encoder outputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from lmdeploy.pytorch.multimodal.data_type import MultiModalData

DEFAULT_EPD_ENCODER_CACHE_BYTES = 4 * 1024**3
EPD_ENCODER_CACHE_MAX_BYTES_ENV = 'LMDEPLOY_EPD_ENCODER_CACHE_MAX_BYTES'


@dataclass
class EPDEncoderCacheEntry:
    """GPU-resident encoder output cached by one multimodal input hash."""

    key: str
    tensor: torch.Tensor
    nbytes: int
    ref_count: int = 0
    mr_key: str | None = None
    on_evict: Callable[[], None] | None = None


class EPDEncoderCache:
    """Byte-limited LRU cache for encoder output tensors."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.entries: OrderedDict[str, EPDEncoderCacheEntry] = OrderedDict()

    @classmethod
    def from_env(cls) -> 'EPDEncoderCache':
        max_bytes = int(os.getenv(EPD_ENCODER_CACHE_MAX_BYTES_ENV, DEFAULT_EPD_ENCODER_CACHE_BYTES))
        return cls(max_bytes=max_bytes)

    def get(self, key: str) -> torch.Tensor | None:
        entry = self.get_entry(key)
        return None if entry is None else entry.tensor

    def get_entry(self, key: str) -> EPDEncoderCacheEntry | None:
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def put(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if self.max_bytes == 0 or nbytes > self.max_bytes:
            return tensor

        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key].tensor

        self._evict_until_fits(nbytes)
        self.entries[key] = EPDEncoderCacheEntry(key=key, tensor=tensor, nbytes=nbytes)
        self.used_bytes += nbytes
        return tensor

    def pin(self, key: str):
        self.entries[key].ref_count += 1

    def unpin(self, key: str):
        entry = self.entries.get(key)
        if entry is not None:
            entry.ref_count = max(0, entry.ref_count - 1)

    def clear(self):
        for entry in list(self.entries.values()):
            self._evict_entry(entry)
        self.entries.clear()
        self.used_bytes = 0

    def _evict_entry(self, entry: EPDEncoderCacheEntry):
        if entry.on_evict is not None:
            entry.on_evict()
            entry.on_evict = None
        entry.mr_key = None

    def _evict_until_fits(self, incoming_bytes: int):
        while self.used_bytes + incoming_bytes > self.max_bytes:
            victim_key = next((key for key, entry in self.entries.items() if entry.ref_count == 0), None)
            if victim_key is None:
                raise RuntimeError('EPD encoder cache is full and all entries are pinned.')
            victim = self.entries.pop(victim_key)
            self._evict_entry(victim)
            self.used_bytes -= victim.nbytes


_EPD_ENCODER_CACHE: EPDEncoderCache | None = None


def get_epd_encoder_cache() -> EPDEncoderCache | None:
    """Return the process-local producer cache, or None when disabled."""
    global _EPD_ENCODER_CACHE
    if _EPD_ENCODER_CACHE is None:
        cache = EPDEncoderCache.from_env()
        if cache.max_bytes == 0:
            return None
        _EPD_ENCODER_CACHE = cache
    return _EPD_ENCODER_CACHE


def set_epd_encoder_cache(cache: EPDEncoderCache | None):
    """Set the process-local cache; intended for lifecycle setup and tests."""
    global _EPD_ENCODER_CACHE
    _EPD_ENCODER_CACHE = cache


def _normalize_cache_meta(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'data': tensor.tolist(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            'dtype': str(array.dtype),
            'shape': list(array.shape),
            'data': array.tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _normalize_cache_meta(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_meta(item) for item in value]
    raise TypeError(f'Unsupported EPD cache metadata value: {type(value)}')


def build_epd_mm_cache_key(mm_input: MultiModalData) -> str:
    """Build a stable hash for one preprocessed multimodal item."""
    hasher = hashlib.sha256()
    hasher.update(str(mm_input.modality.value).encode())
    tensors = [mm_input.data] if isinstance(mm_input.data, torch.Tensor) else mm_input.data
    for tensor in tensors:
        tensor = tensor.detach().contiguous().cpu()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    meta = _normalize_cache_meta(mm_input.meta)
    hasher.update(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
    mrope_pos_ids = _normalize_cache_meta(mm_input.mrope_pos_ids)
    hasher.update(json.dumps(mrope_pos_ids, sort_keys=True, separators=(',', ':')).encode())
    return hasher.hexdigest()
