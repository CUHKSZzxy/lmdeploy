# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for the first EPD language-side receive path."""

import inspect

import numpy as np
import torch

from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, EncoderHttpJsonEmbedding, MigrationProtocol
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.serve.epd_channel import (
    EPD_BACKEND_HTTP_JSON,
    EPD_BACKEND_ZMQ_IPC,
    EPD_TRANSFER_BACKENDS,
    EncoderTransferEmbedding,
    EncoderTransferPayload,
    recv_epd_payload,
    send_epd_payload,
)

_NUMPY_DTYPES = {
    None: np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': np.float32,
}


def _as_numpy_dtype(dtype: str | None):
    """Map serialized embedding dtype names to numpy dtypes."""
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f'unsupported HTTP JSON encoder embedding dtype: {dtype}')
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


def _embedding_payloads(prompt_input: dict) -> tuple[list[EncoderTransferEmbedding], list[list[int]], list[list[int]],
                                                    list[str]]:
    input_embeddings = prompt_input.get('input_embeddings') or []
    input_embedding_ranges = prompt_input.get('input_embedding_ranges')
    payload_embeddings: list[EncoderTransferEmbedding] = []
    shapes: list[list[int]] = []
    dtypes: list[str] = []

    for index, embedding in enumerate(input_embeddings):
        start, end = _embedding_range(embedding, input_embedding_ranges, index)
        data, _ = _embedding_array_and_dtype(embedding)
        if data.ndim != 2:
            raise ValueError(f'encoder embedding must be 2-D, got shape {data.shape}')
        if end - start != data.shape[0]:
            raise ValueError(f'encoder embedding range [{start}, {end}) does not match embedding rows {data.shape[0]}')
        data = np.ascontiguousarray(data)
        dtype = str(data.dtype)
        payload_embeddings.append(EncoderTransferEmbedding(data=data, start=start, end=end, dtype=dtype))
        shapes.append(list(data.shape))
        dtypes.append(dtype)

    ranges = [[emb.start, emb.end] for emb in payload_embeddings]
    return payload_embeddings, ranges, shapes, dtypes


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

    raise ValueError('EPD encoder embedding computation currently requires a local PyTorch model.')


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
        raise ValueError('EPD encoder embedding computation requires a model visual encoder.')
    if not all(hasattr(visual, attr) for attr in ('rot_pos_emb', 'fast_pos_embed_interpolate')):
        raise ValueError('EPD encoder embedding computation requires a Qwen3.5-style visual encoder.')
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


def compute_encoder_prompt_input(prompt_input: dict, model_or_engine) -> dict:
    """Run the non-DeepStack visual path and attach final image embeddings.

    This turns the PyTorch Qwen3.5-style ``input_ids + multimodal`` prompt
    input into ``input_ids + input_embeddings`` so the existing HTTP JSON
    ``EncoderCacheRef`` transport can carry the encoder output.
    """
    if prompt_input.get('input_embeddings') or not prompt_input.get('multimodal'):
        return prompt_input

    model = _resolve_encoder_model(model_or_engine)
    _reject_deepstack_model(model)
    visual = _get_visual_module(model)
    input_processor = model.get_input_processor()
    if input_processor is None:
        raise ValueError('EPD encoder embedding computation requires a model input processor.')

    input_ids = _to_int_list(prompt_input['input_ids'])
    processed = input_processor.preprocess_input(input_ids, prompt_input['multimodal'])
    input_ids = _to_int_list(processed.input_ids)
    input_multimodals = processed.input_multimodals or {}
    mm_inputs = input_multimodals.get('mm_data', [])
    if not mm_inputs:
        return dict(prompt_input, input_ids=input_ids)
    if any(not _is_visual_modality(mm_input) for mm_input in mm_inputs):
        raise ValueError('EPD encoder embedding computation currently supports image/video visual inputs only.')

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


async def compute_encoder_prompt_input_for_engine(prompt_input: dict, model_or_engine) -> dict:
    """Compute encoder embeddings for a prompt input, delegating to MP worker if needed."""
    engine = getattr(model_or_engine, 'engine', None)
    remote_compute = getattr(engine, 'compute_encoder_prompt_input', None)
    if callable(remote_compute):
        computed = remote_compute(prompt_input)
        if inspect.isawaitable(computed):
            computed = await computed
        return computed
    return compute_encoder_prompt_input(prompt_input, model_or_engine)


def encoder_cache_ref_to_prompt_input(encoder_result: EncoderCacheRef) -> dict:
    """Convert an EPD encoder cache reference into AsyncEngine prompt input.

    The first bring-up path supports HTTP JSON embeddings so the language engine
    can exercise the existing ``input_embeddings`` model-input plumbing before
    a real remote cache-transfer backend is added.
    """
    if encoder_result.backend != EPD_BACKEND_HTTP_JSON:
        raise ValueError(f'EPD backend {encoder_result.backend!r} requires async channel receive.')

    prompt_input = dict(prompt=None, input_ids=list(encoder_result.token_ids))

    if not encoder_result.input_embeddings:
        return prompt_input

    embeddings = []
    for http_json_embedding in encoder_result.input_embeddings:
        dtype = _as_numpy_dtype(http_json_embedding.dtype or encoder_result.dtype)
        data = np.asarray(http_json_embedding.data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError(f'HTTP JSON encoder embedding must be 2-D, got shape {data.shape}')
        expected_rows = http_json_embedding.end - http_json_embedding.start
        if expected_rows != data.shape[0]:
            raise ValueError(
                f'HTTP JSON encoder embedding range [{http_json_embedding.start}, {http_json_embedding.end}) '
                f'does not match embedding rows {data.shape[0]}')
        embeddings.append(InputEmbeddings(data, start=http_json_embedding.start, end=http_json_embedding.end))

    prompt_input['input_embeddings'] = embeddings
    return prompt_input


async def encoder_cache_ref_to_prompt_input_async(encoder_result: EncoderCacheRef) -> dict:
    """Convert an encoder cache ref into prompt input, receiving channel data if needed."""
    if encoder_result.backend == EPD_BACKEND_HTTP_JSON:
        return encoder_cache_ref_to_prompt_input(encoder_result)
    if encoder_result.backend != EPD_BACKEND_ZMQ_IPC:
        raise ValueError(f'unsupported EPD encoder transfer backend: {encoder_result.backend}')
    if not encoder_result.transfer_id:
        raise ValueError('EPD channel encoder_result requires transfer_id')

    payload = await recv_epd_payload(encoder_result.transfer_id)
    if payload.token_ids and payload.token_ids != list(encoder_result.token_ids):
        raise ValueError('EPD channel payload token_ids do not match encoder_result')

    prompt_input = dict(prompt=None, input_ids=list(encoder_result.token_ids))
    embeddings = []
    for embedding in payload.embeddings:
        expected_rows = embedding.end - embedding.start
        if expected_rows != embedding.data.shape[0]:
            raise ValueError(
                f'EPD channel embedding range [{embedding.start}, {embedding.end}) '
                f'does not match embedding rows {embedding.data.shape[0]}')
        embeddings.append(InputEmbeddings(embedding.data, start=embedding.start, end=embedding.end))
    if embeddings:
        prompt_input['input_embeddings'] = embeddings
    return prompt_input


async def send_prompt_input_via_channel(prompt_input: dict, transfer_id: str, channel_address: str):
    """Send computed encoder embeddings through the configured EPD channel."""
    if not transfer_id:
        raise ValueError('EPD channel transfer requires transfer_id')
    if not channel_address:
        raise ValueError('EPD channel transfer requires channel_address')
    embeddings, _, _, _ = _embedding_payloads(prompt_input)
    if not embeddings:
        raise ValueError('EPD channel transfer requires precomputed input_embeddings')
    payload = EncoderTransferPayload(
        transfer_id=transfer_id,
        token_ids=_to_int_list(prompt_input.get('input_ids') or []),
        embeddings=embeddings,
    )
    await send_epd_payload(channel_address, payload)


def prompt_input_to_encoder_cache_ref(prompt_input: dict,
                                      remote_engine_id: str,
                                      remote_session_id: int,
                                      protocol: MigrationProtocol = MigrationProtocol.TCP,
                                      backend: str = EPD_BACKEND_HTTP_JSON,
                                      transfer_id: str | None = None,
                                      channel_address: str | None = None) -> EncoderCacheRef:
    """Serialize prompt-side embeddings into an EPD encoder cache ref.

    This is a bring-up producer for encoder outputs that already exist as
    ``input_embeddings``. PyTorch VLM prompt inputs that still contain raw
    pixel-value multimodal data need a model-specific vision-forward producer
    before they can use this helper.
    """
    if backend not in EPD_TRANSFER_BACKENDS:
        raise ValueError(f'unsupported EPD encoder transfer backend: {backend}')
    input_ids = _to_int_list(prompt_input.get('input_ids') or [])
    input_embeddings = prompt_input.get('input_embeddings')
    if not input_embeddings:
        if prompt_input.get('multimodal') or prompt_input.get('input_multimodals'):
            raise ValueError('EPD encoder producer requires precomputed input_embeddings.')
        return EncoderCacheRef(token_ids=input_ids,
                               protocol=protocol,
                               backend=backend,
                               transfer_id=transfer_id,
                               channel_address=channel_address,
                               remote_engine_id=remote_engine_id,
                               remote_session_id=remote_session_id,
                               remote_block_ids=[])

    payload_embeddings, ranges, shapes, dtypes = _embedding_payloads(prompt_input)
    http_json_embeddings = None
    if backend == EPD_BACKEND_HTTP_JSON:
        http_json_embeddings = [
            EncoderHttpJsonEmbedding(data=embedding.data.astype(np.float32).tolist(),
                                     start=embedding.start,
                                     end=embedding.end,
                                     dtype=embedding.dtype) for embedding in payload_embeddings
        ]
    elif not transfer_id:
        raise ValueError('non-http_json EPD encoder_result requires transfer_id')

    return EncoderCacheRef(token_ids=input_ids,
                           input_embedding_ranges=ranges,
                           input_embeddings=http_json_embeddings,
                           protocol=protocol,
                           backend=backend,
                           transfer_id=transfer_id,
                           channel_address=channel_address,
                           remote_engine_id=remote_engine_id,
                           remote_session_id=remote_session_id,
                           remote_block_ids=[],
                           dtype=dtypes[0] if len(set(dtypes)) == 1 else None,
                           shape=shapes)
