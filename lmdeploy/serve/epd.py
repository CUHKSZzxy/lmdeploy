# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for the first EPD language-side receive path."""

import numpy as np

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef
from lmdeploy.pytorch.messages import InputEmbeddings

_NUMPY_DTYPES = {
    None: np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': np.float32,
}


def _as_numpy_dtype(dtype: str | None):
    """Map serialized embedding dtype names to numpy dtypes."""
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f'unsupported inline encoder embedding dtype: {dtype}')
    return _NUMPY_DTYPES[dtype]


def encoder_cache_ref_to_prompt_input(encoder_result: EncoderCacheRef) -> dict:
    """Convert an EPD encoder cache reference into AsyncEngine prompt input.

    The first bring-up path supports inline embeddings so the language engine
    can exercise the existing ``input_embeddings`` model-input plumbing before
    a real remote cache-transfer backend is added.
    """
    prompt_input = dict(prompt=None, input_ids=list(encoder_result.token_ids))

    if not encoder_result.input_embeddings:
        return prompt_input

    embeddings = []
    for inline_embedding in encoder_result.input_embeddings:
        dtype = _as_numpy_dtype(inline_embedding.dtype or encoder_result.dtype)
        data = np.asarray(inline_embedding.data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError(f'inline encoder embedding must be 2-D, got shape {data.shape}')
        expected_rows = inline_embedding.end - inline_embedding.start
        if expected_rows != data.shape[0]:
            raise ValueError(
                f'inline encoder embedding range [{inline_embedding.start}, {inline_embedding.end}) '
                f'does not match embedding rows {data.shape[0]}')
        embeddings.append(InputEmbeddings(data, start=inline_embedding.start, end=inline_embedding.end))

    prompt_input['input_embeddings'] = embeddings
    return prompt_input
