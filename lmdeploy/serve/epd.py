# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for the first EPD language-side receive path."""

import inspect

import numpy as np
import torch

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


def _unwrap_encoder_model(model, _visited=None):
    if _visited is None:
        _visited = set()
    if model is None or id(model) in _visited:
        return None
    _visited.add(id(model))

    if hasattr(model, 'get_input_embeddings') and hasattr(model, 'get_input_processor'):
        return model

    get_model = getattr(model, 'get_model', None)
    if callable(get_model):
        unwrapped = _unwrap_encoder_model(get_model(), _visited)
        if unwrapped is not None:
            return unwrapped

    for attr in ('model', 'module', 'patched_model'):
        unwrapped = _unwrap_encoder_model(getattr(model, attr, None), _visited)
        if unwrapped is not None:
            return unwrapped

    return None


def _resolve_encoder_model(model_or_engine):
    candidates = [model_or_engine]
    engine = getattr(model_or_engine, 'engine', None)
    if engine is not None:
        candidates.append(engine)
    executor = getattr(engine or model_or_engine, 'executor', None)
    model_agent = getattr(executor, 'model_agent', None)
    candidates.append(getattr(model_agent, 'patched_model', None))

    for candidate in candidates:
        model = _unwrap_encoder_model(candidate)
        if model is not None:
            return model

    raise ValueError('EPD encoder materialization currently requires a local PyTorch model.')


def _get_visual_module(model, _visited=None):
    if _visited is None:
        _visited = set()
    if model is None or id(model) in _visited:
        return None
    _visited.add(id(model))

    visual = getattr(model, 'visual', None)
    if visual is None:
        for attr in ('model', 'module', 'patched_model'):
            visual = _get_visual_module(getattr(model, attr, None), _visited)
            if visual is not None:
                break
    if visual is None:
        raise ValueError('EPD encoder materialization requires a model visual encoder.')
    if not all(hasattr(visual, attr) for attr in ('rot_pos_emb', 'fast_pos_embed_interpolate')):
        raise ValueError('EPD encoder materialization requires a Qwen3.5-style visual encoder.')
    return visual


def _reject_deepstack_model(model):
    config = getattr(model, 'config', None)
    vision_config = getattr(config, 'vision_config', None)
    deepstack_indexes = getattr(vision_config, 'deepstack_visual_indexes', None)
    if deepstack_indexes:
        raise ValueError('DeepStack visual embeddings are not supported by the first EPD encoder producer.')


def _module_device_and_dtype(module, fallback_device=None, fallback_dtype=None):
    parameter = next(module.parameters(), None)
    device = getattr(parameter, 'device', fallback_device)
    dtype = getattr(parameter, 'dtype', fallback_dtype)
    return device or torch.device('cpu'), dtype or torch.float32


def _is_visual_modality(mm_input) -> bool:
    modality = getattr(mm_input, 'modality', None)
    modality_name = getattr(modality, 'name', str(modality))
    return modality_name in ('IMAGE', 'VIDEO', 'Modality.IMAGE', 'Modality.VIDEO')


def _has_deepstack_payload(payload) -> bool:
    if payload is None:
        return False
    if isinstance(payload, (list, tuple)):
        return len(payload) > 0
    return True


def materialize_encoder_prompt_input(prompt_input: dict, model_or_engine) -> dict:
    """Run the non-DeepStack visual path and attach final image embeddings.

    This turns the PyTorch Qwen3.5-style ``input_ids + multimodal`` prompt
    input into ``input_ids + input_embeddings`` so the existing inline
    ``EncoderCacheRef`` transport can carry the encoder output.
    """
    if prompt_input.get('input_embeddings') or not prompt_input.get('multimodal'):
        return prompt_input

    model = _resolve_encoder_model(model_or_engine)
    _reject_deepstack_model(model)
    visual = _get_visual_module(model)
    input_processor = model.get_input_processor()
    if input_processor is None:
        raise ValueError('EPD encoder materialization requires a model input processor.')

    input_ids = _to_int_list(prompt_input['input_ids'])
    processed = input_processor.preprocess_input(input_ids, prompt_input['multimodal'])
    input_ids = _to_int_list(processed.input_ids)
    input_multimodals = processed.input_multimodals or {}
    mm_inputs = input_multimodals.get('mm_data', [])
    if not mm_inputs:
        return dict(prompt_input, input_ids=input_ids)
    if any(not _is_visual_modality(mm_input) for mm_input in mm_inputs):
        raise ValueError('EPD encoder materialization currently supports image/video visual inputs only.')

    embedding_layer = model.get_input_embeddings()
    embed_device, embed_dtype = _module_device_and_dtype(embedding_layer)
    visual_device, _ = _module_device_and_dtype(visual, fallback_device=embed_device, fallback_dtype=embed_dtype)
    pixel_values = torch.cat([mm_input.data for mm_input in mm_inputs]).to(device=visual_device, dtype=embed_dtype)
    grid_thw = torch.stack([torch.as_tensor(mm_input.meta['grid_thw'], dtype=torch.long) for mm_input in mm_inputs])

    vis_pos_emb = visual.rot_pos_emb(grid_thw)
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    vis_cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).to(pixel_values.device)
    vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
    vis_pos_emb = vis_pos_emb.repeat(1, 2)
    vis_pos_emb = (vis_pos_emb.cos().to(pixel_values.device), vis_pos_emb.sin().to(pixel_values.device))

    with torch.inference_mode():
        image_embeds = visual(pixel_values,
                              cu_seqlens=vis_cu_seqlens,
                              rotary_pos_emb=vis_pos_emb,
                              pos_embeds=pos_embeds)
    if isinstance(image_embeds, tuple):
        if len(image_embeds) > 1 and _has_deepstack_payload(image_embeds[1]):
            raise ValueError('DeepStack visual embeddings are not supported by the first EPD encoder producer.')
        image_embeds = image_embeds[0]

    merge_size = getattr(visual, 'spatial_merge_size', 1)
    split_sizes = (grid_thw.prod(-1) // merge_size**2).tolist()
    split_embeddings = torch.split(image_embeds, split_sizes)

    input_embeddings = []
    input_embedding_ranges = []
    for embedding, mm_input in zip(split_embeddings, mm_inputs):
        start, end = int(mm_input.start), int(mm_input.end)
        if end - start != embedding.shape[0]:
            raise ValueError(f'encoder embedding range [{start}, {end}) does not match embedding rows '
                             f'{embedding.shape[0]}')
        embedding = embedding.to(device=embed_device, dtype=embed_dtype).detach().float().cpu().numpy()
        input_embeddings.append(InputEmbeddings(embedding, start=start, end=end))
        input_embedding_ranges.append([start, end])

    prompt_input = dict(prompt_input)
    prompt_input.pop('multimodal', None)
    prompt_input['input_ids'] = input_ids
    prompt_input['input_embeddings'] = input_embeddings
    prompt_input['input_embedding_ranges'] = input_embedding_ranges
    return prompt_input


async def materialize_encoder_prompt_input_for_engine(prompt_input: dict, model_or_engine) -> dict:
    """Materialize encoder prompt input, delegating to MP worker if needed."""
    engine = getattr(model_or_engine, 'engine', None)
    remote_materializer = getattr(engine, 'materialize_encoder_prompt_input', None)
    if callable(remote_materializer):
        materialized = remote_materializer(prompt_input)
        if inspect.isawaitable(materialized):
            materialized = await materialized
        return materialized
    return materialize_encoder_prompt_input(prompt_input, model_or_engine)


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
