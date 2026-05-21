# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for the first EPD language-side receive path."""

import numpy as np

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, EncoderInlineEmbedding, MigrationProtocol
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


def _to_int_list(values) -> list[int]:
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return [int(value) for value in values]


def _embedding_array_and_dtype(embedding) -> tuple[np.ndarray, str]:
    """Convert one embedding payload to a numpy array and serialize its
    dtype."""
    data = getattr(embedding, 'embeddings', embedding)
    dtype = str(getattr(data, 'dtype', 'float32'))
    if dtype.startswith('torch.'):
        dtype = dtype.split('.', 1)[1]
    if hasattr(data, 'detach'):
        if dtype == 'bfloat16':
            data = data.float()
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)
        dtype = str(data.dtype)
    return np.asarray(data), dtype


def _embedding_range(embedding, ranges, index: int) -> list[int]:
    if ranges is not None:
        return _to_int_list(ranges[index])
    if hasattr(embedding, 'start') and hasattr(embedding, 'end'):
        return [int(embedding.start), int(embedding.end)]
    raise ValueError('input_embedding_ranges are required to serialize encoder embeddings.')


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


def prompt_input_to_encoder_cache_ref(prompt_input: dict,
                                      remote_engine_id: str,
                                      remote_session_id: int,
                                      protocol: MigrationProtocol = MigrationProtocol.TCP) -> EncoderCacheRef:
    """Serialize prompt-side inline embeddings into an EPD encoder cache ref.

    This is a bring-up producer for encoder outputs that already exist as
    ``input_embeddings``. PyTorch VLM prompt inputs that still contain raw
    pixel-value multimodal data need a model-specific vision-forward producer
    before they can use this helper.
    """
    input_ids = _to_int_list(prompt_input.get('input_ids') or [])
    input_embeddings = prompt_input.get('input_embeddings')
    if not input_embeddings:
        if prompt_input.get('multimodal') or prompt_input.get('input_multimodals'):
            raise ValueError('EPD inline encoder producer requires precomputed input_embeddings.')
        return EncoderCacheRef(token_ids=input_ids,
                               protocol=protocol,
                               backend='inline',
                               remote_engine_id=remote_engine_id,
                               remote_session_id=remote_session_id,
                               remote_block_ids=[])

    input_embedding_ranges = prompt_input.get('input_embedding_ranges')
    inline_embeddings: list[EncoderInlineEmbedding] = []
    shapes: list[list[int]] = []
    dtypes: list[str] = []

    for index, embedding in enumerate(input_embeddings):
        start, end = _embedding_range(embedding, input_embedding_ranges, index)
        data, dtype = _embedding_array_and_dtype(embedding)
        if data.ndim != 2:
            raise ValueError(f'encoder embedding must be 2-D, got shape {data.shape}')
        if end - start != data.shape[0]:
            raise ValueError(f'encoder embedding range [{start}, {end}) does not match embedding rows {data.shape[0]}')
        inline_embeddings.append(
            EncoderInlineEmbedding(data=data.astype(np.float32).tolist(), start=start, end=end, dtype=dtype))
        shapes.append(list(data.shape))
        dtypes.append(dtype)

    return EncoderCacheRef(token_ids=input_ids,
                           input_embedding_ranges=[[emb.start, emb.end] for emb in inline_embeddings],
                           input_embeddings=inline_embeddings,
                           protocol=protocol,
                           backend='inline',
                           remote_engine_id=remote_engine_id,
                           remote_session_id=remote_session_id,
                           remote_block_ids=[],
                           dtype=dtypes[0] if len(set(dtypes)) == 1 else None,
                           shape=shapes)
