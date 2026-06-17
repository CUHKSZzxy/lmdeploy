# Copyright (c) OpenMMLab. All rights reserved.
"""Producer-side GPU cache for EPD encoder outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.multimodal.data_type import MultiModalData, make_multimodal_content_hash


@dataclass
class EncoderCacheEntry:
    """GPU-resident encoder output cached by one multimodal input hash."""

    key: str
    tensor: torch.Tensor
    nbytes: int
    ref_count: int = 0
    mr_key: str | None = None
    on_evict: Callable[[], None] | None = None


class EncoderCache:
    """Byte-limited LRU cache for encoder output tensors."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.entries: OrderedDict[str, EncoderCacheEntry] = OrderedDict()

    @classmethod
    def from_config(cls, config) -> 'EncoderCache | None':
        cache_size_gb = config.encoder_cache_size_gb
        if cache_size_gb <= 0:
            return None
        return cls(max_bytes=int(cache_size_gb * 1024**3))

    def get(self, key: str) -> torch.Tensor | None:
        entry = self.get_entry(key)
        return None if entry is None else entry.tensor

    def get_entry(self, key: str) -> EncoderCacheEntry | None:
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
        self.entries[key] = EncoderCacheEntry(key=key, tensor=tensor, nbytes=nbytes)
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

    def _evict_entry(self, entry: EncoderCacheEntry):
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


_EPD_ENCODER_CACHE: EncoderCache | None = None


def get_epd_encoder_cache() -> EncoderCache | None:
    """Return the process-local producer cache, or None when disabled."""
    return _EPD_ENCODER_CACHE


def set_epd_encoder_cache(cache: EncoderCache | None):
    """Set the process-local cache; intended for lifecycle setup and tests."""
    global _EPD_ENCODER_CACHE
    _EPD_ENCODER_CACHE = cache


def set_epd_encoder_cache_from_config(config):
    """Set the process-local cache from engine configuration."""
    set_epd_encoder_cache(EncoderCache.from_config(config))


def build_epd_mm_cache_key(mm_input: MultiModalData) -> str:
    """Build a stable hash for one preprocessed multimodal item."""
    content_hash = mm_input.content_hash
    if content_hash is None:
        content_hash = make_multimodal_content_hash(mm_input.data, mm_input.meta, mm_input.mrope_pos_ids)
        mm_input.content_hash = content_hash
    return f'{mm_input.modality.value}:{content_hash}'
