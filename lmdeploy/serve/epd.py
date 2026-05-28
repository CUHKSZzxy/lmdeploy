# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for EPD encoder prompt dispatch."""

import inspect


async def compute_encoder_prompt_input_for_engine(prompt_input: dict, model_or_engine) -> dict:
    """Compute encoder embeddings through the PyTorch engine."""
    if prompt_input.get('input_embeddings') or not prompt_input.get('multimodal'):
        return prompt_input

    engine = getattr(model_or_engine, 'engine', model_or_engine)
    computed = engine.compute_encoder_prompt_input(prompt_input)
    if inspect.isawaitable(computed):
        computed = await computed
    return computed
