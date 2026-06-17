# Copyright (c) OpenMMLab. All rights reserved.
"""Model-side helpers for EPD encoder computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lmdeploy.pytorch.disagg.epd.cache import build_epd_mm_cache_key, get_epd_encoder_cache
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.pytorch.multimodal.data_type import MultiModalData


@dataclass
class EPDEncoderItem:
    """One encoder output item and its target token range."""

    embedding: torch.Tensor
    start: int
    end: int
    cache_key: str | None = None


class EPDEncoderMixin:
    """Generic prompt-input flow for EPD encoder producers."""

    def compute_encoder_prompt_input(self, prompt_input: dict) -> dict:
        """Compute encoder embeddings through model-specific hooks."""
        if prompt_input.get('input_embeddings') or not prompt_input.get('multimodal'):
            return prompt_input
        if getattr(self, 'language_only', False):
            raise ValueError(f'{type(self).__name__} cannot compute EPD encoder embeddings in language-only mode.')

        input_ids = prompt_input['input_ids']
        processed = self.input_processor.preprocess_input(input_ids, prompt_input['multimodal'])
        input_ids = processed.input_ids
        input_multimodals = processed.input_multimodals or {}
        mm_inputs = input_multimodals.get('mm_data', [])
        if not mm_inputs:
            return dict(prompt_input, input_ids=input_ids)

        encoder_items = self._compute_epd_encoder_items_with_cache(mm_inputs)

        output = dict(prompt_input)
        output.pop('multimodal', None)
        output['input_ids'] = input_ids
        output['input_embeddings'] = [
            InputEmbeddings(item.embedding, start=item.start, end=item.end) for item in encoder_items
        ]
        output['input_embedding_ranges'] = [[item.start, item.end] for item in encoder_items]
        output['epd_encoder_cache_keys'] = [item.cache_key for item in encoder_items]
        return output

    def _compute_epd_encoder_items_with_cache(self, mm_inputs: list[MultiModalData]) -> list[EPDEncoderItem]:
        cache = get_epd_encoder_cache()
        cache_keys = [build_epd_mm_cache_key(mm_input) if cache is not None else None for mm_input in mm_inputs]
        outputs: list[EPDEncoderItem | None] = [None] * len(mm_inputs)
        miss_indices: list[int] = []

        for index, cache_key in enumerate(cache_keys):
            if cache is not None and cache_key is not None:
                embedding = cache.get(cache_key)
                if embedding is not None:
                    outputs[index] = EPDEncoderItem(
                        embedding=embedding,
                        start=int(mm_inputs[index].start),
                        end=int(mm_inputs[index].end),
                        cache_key=cache_key,
                    )
                    continue
            miss_indices.append(index)

        if miss_indices:
            miss_inputs = [mm_inputs[index] for index in miss_indices]
            miss_outputs = self._compute_epd_encoder_items_uncached(miss_inputs)
            for index, item in zip(miss_indices, miss_outputs):
                cache_key = cache_keys[index]
                if cache is not None and cache_key is not None:
                    try:
                        item.embedding = cache.put(cache_key, item.embedding)
                        if cache.get_entry(cache_key) is not None:
                            item.cache_key = cache_key
                    except RuntimeError:
                        item.cache_key = None
                outputs[index] = item

        return outputs

    def _compute_epd_encoder_items_uncached(self, mm_inputs: list[MultiModalData]) -> list[EPDEncoderItem]:
        raise NotImplementedError
