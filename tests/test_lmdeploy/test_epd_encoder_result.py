import asyncio
import contextlib
import ctypes
import importlib
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from lmdeploy.archs import get_task
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, ResponseType
from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.pytorch.disagg.conn.protocol import (
    EPDConnectionRequest,
    EPDConnectionStatus,
    EPDDropConnectionRequest,
    EPDInitRequest,
    EncoderOutputMetadata,
    EncoderOutputRef,
    EncoderTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.epd.cache import (
    EncoderCache,
    build_epd_mm_cache_key,
    get_epd_encoder_cache,
    set_epd_encoder_cache,
    set_epd_encoder_cache_from_config,
)
from lmdeploy.pytorch.disagg.epd.control import (
    EncoderTransferConfig,
    build_encoder_transfer_config,
)
from lmdeploy.pytorch.disagg.epd.manager import (
    EncoderTransferManager,
    _build_embedding_layout,
)
from lmdeploy.pytorch.disagg.epd.engine import (
    compute_encoder_prompt_input_for_engine,
)
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.executor.base import ExecutorBase
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.input_process import PreprocessInputResult
from lmdeploy.pytorch.engine.request import RequestType, Response
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from lmdeploy.pytorch.models.utils.epd import EPDEncoderMixin
from lmdeploy.pytorch.models.utils.model import DeployModelMixinV1, build_language_model
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.serve.core import AsyncEngine
from lmdeploy.vl.constants import Modality

async_engine_mod = importlib.import_module('lmdeploy.serve.core.async_engine')


@pytest.fixture(autouse=True)
def _reset_epd_encoder_cache():
    set_epd_encoder_cache(None)
    yield
    set_epd_encoder_cache(None)


def test_engine_config_rejects_language_only_with_encoder_only():
    with pytest.raises(ValueError, match='language_only and encoder_only'):
        PytorchEngineConfig(language_only=True, encoder_only=True)


def test_get_task_uses_language_only_without_legacy_disable_flag():
    task, pipeline_class = get_task('pytorch', 'unused', backend_config=PytorchEngineConfig(language_only=True))

    assert task == 'llm'
    assert pipeline_class is AsyncEngine


def test_async_engine_removes_session_when_encoder_output_ref_load_fails(monkeypatch):

    class _FakeSession:
        session_id = 7
        step = 0

    class _FakeSessionManager:

        def __init__(self):
            self.session = _FakeSession()
            self.removed = []

        def get(self, session_id, step=0):
            return self.session

        def remove(self, session):
            self.removed.append(session)

    async def _fail_load_encoder_output(encoder_output_ref):
        raise ValueError('stale encoder output')

    monkeypatch.setattr(async_engine_mod, 'load_encoder_output_async', _fail_load_encoder_output)

    session_mgr = _FakeSessionManager()
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = session_mgr
    engine.request_logger = object()
    encoder_output_ref = EncoderOutputRef(
        token_ids=[1, 2],
        input_embedding_ranges=[[0, 2]],
        transfer_id='epd-test',
        protocol=MigrationProtocol.RDMA,
        remote_engine_id='http://encoder',
        remote_session_id=3,
        dtype='float32',
        shape=[[2, 4]],
        transfer_metadata=EncoderOutputMetadata(endpoint_info={'name': 'encoder'}),
    )

    async def _run():
        return [
            output async for output in engine.generate(
                messages=[{'role': 'user', 'content': 'hello'}],
                session_id=7,
                gen_config=GenerationConfig(max_new_tokens=1, encoder_output_ref=encoder_output_ref),
            )
        ]

    outputs = asyncio.run(_run())

    assert outputs[0].finish_reason == 'error'
    assert session_mgr.removed == [session_mgr.session]


def test_build_language_model_skips_module_in_encoder_only_context():

    class _TinyLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    with build_model_context(BuildModelContext(encoder_only=True)):
        model = build_language_model(_TinyLanguageModel)

    assert isinstance(model, nn.Identity)
    assert model._is_dummy_mod


def test_build_lm_head_skips_module_in_encoder_only_context():
    with build_model_context(BuildModelContext(encoder_only=True)):
        lm_head = DeployModelMixinV1().build_lm_head(hidden_size=1, vocab_size=1)

    assert lm_head is None


def test_executor_caps_cache_config_for_encoder_only():

    class _Executor(ExecutorBase):

        def gather_free_mem(self):
            raise AssertionError('encoder-only config should not size cache from free memory')

        def set_cache_config(self, cache_config, spec_cache_config=None):
            self.updated_cache_config = cache_config
            self.updated_spec_cache_config = spec_cache_config

        def set_model_config(self, model_config, spec_model_config=None):
            self.updated_model_config = model_config
            self.updated_spec_model_config = spec_model_config

    model_config = types.SimpleNamespace(
        sliding_window=-1,
        k_head_dim=128,
        states_shapes=[((4, 8), torch.float32)],
        use_flash_mla=False,
    )
    cache_config = CacheConfig(
        max_batches=32,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=0,
        kernel_block_size=64,
        max_prefill_token_num=8192,
        num_reserved_gpu_blocks=1,
        role=EngineRole.Encoder,
    )
    executor = _Executor(
        'unused',
        model_config=model_config,
        cache_config=cache_config,
        backend_config=types.SimpleNamespace(),
        dist_config=types.SimpleNamespace(dp=1, world_size=1),
        misc_config=types.SimpleNamespace(encoder_only=True),
    )

    executor.update_configs()

    assert cache_config.num_gpu_blocks == 3
    assert cache_config.max_prefill_token_num == 64
    assert cache_config.states_shapes == []
    assert cache_config.num_state_caches == 0
    assert executor.updated_cache_config is cache_config
    assert executor.updated_spec_cache_config is None
    assert executor.updated_model_config is model_config
    assert executor.updated_spec_model_config is None


def test_encoder_transfer_config_generates_request_fields():
    config = build_encoder_transfer_config(receiver_engine_id='http://language')

    assert config.transfer_id.startswith('epd-')
    assert config.to_request_fields() == {
        'epd_transfer_id': config.transfer_id,
        'encoder_output_receiver_engine_id': 'http://language',
    }


def test_encoder_transfer_config_requires_transfer_id_and_receiver_engine():
    with pytest.raises(ValueError, match='requires transfer_id'):
        EncoderTransferConfig(receiver_engine_id='http://language')

    with pytest.raises(ValueError, match='requires receiver_engine_id'):
        EncoderTransferConfig(transfer_id='epd-test')


def test_build_embedding_layout_preserves_ranges_and_rejects_mismatch():
    first = InputEmbeddings(torch.ones(2, 4), start=3, end=5)
    second = InputEmbeddings(torch.ones(3, 4), start=10, end=13)

    layout = _build_embedding_layout({'input_embeddings': [first, second]}, torch.device('cpu'))

    assert layout.tensor.is_contiguous()
    assert layout.tensor.shape == (5, 4)
    assert layout.ranges == [[3, 5], [10, 13]]
    assert layout.shapes == [[2, 4], [3, 4]]
    assert layout.nbytes == layout.tensor.numel() * layout.tensor.element_size()

    mismatch = InputEmbeddings(torch.ones(2, 4), start=3, end=6)
    with pytest.raises(ValueError, match='does not match rows'):
        _build_embedding_layout({'input_embeddings': [mismatch]}, torch.device('cpu'))


def test_epd_encoder_cache_is_configured_explicitly():
    assert PytorchEngineConfig().encoder_cache_size_gb == 4.0
    assert PytorchEngineConfig(encoder_cache_size_gb=0).encoder_cache_size_gb == 0

    with pytest.raises(AssertionError, match='invalid encoder_cache_size_gb'):
        PytorchEngineConfig(encoder_cache_size_gb=-1)

    config = PytorchEngineConfig()
    set_epd_encoder_cache_from_config(config)
    assert get_epd_encoder_cache().max_bytes == int(config.encoder_cache_size_gb * 1024**3)

    set_epd_encoder_cache_from_config(PytorchEngineConfig(encoder_cache_size_gb=0))
    assert get_epd_encoder_cache() is None


def test_epd_mm_cache_key_tracks_tensor_and_meta():
    mm_input = MultiModalData(
        modality=Modality.IMAGE,
        data=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        start=0,
        end=1,
        meta={'grid_thw': torch.tensor([1, 1, 1])},
    )
    same = MultiModalData(
        modality=Modality.IMAGE,
        data=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        start=5,
        end=6,
        meta={'grid_thw': torch.tensor([1, 1, 1])},
    )
    changed = MultiModalData(
        modality=Modality.IMAGE,
        data=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        start=0,
        end=1,
        meta={'grid_thw': torch.tensor([1, 1, 1])},
    )

    assert build_epd_mm_cache_key(mm_input) == build_epd_mm_cache_key(same)
    assert build_epd_mm_cache_key(mm_input) != build_epd_mm_cache_key(changed)
    assert mm_input.content_hash is not None

    same_content_audio = MultiModalData(
        modality=Modality.AUDIO,
        data=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        start=0,
        end=1,
        meta={'grid_thw': torch.tensor([1, 1, 1])},
    )
    same_content_audio.content_hash = mm_input.content_hash
    assert build_epd_mm_cache_key(mm_input) != build_epd_mm_cache_key(same_content_audio)


def test_epd_mm_cache_key_supports_bfloat16_tensor():
    mm_input = MultiModalData(
        modality=Modality.IMAGE,
        data=torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        start=0,
        end=1,
        meta={'grid_thw': torch.tensor([1, 1, 1])},
    )

    assert build_epd_mm_cache_key(mm_input) is not None


def test_epd_encoder_cache_evicts_only_unpinned_entries():
    cache = EncoderCache(max_bytes=32)
    first = torch.ones(2, 2, dtype=torch.float32)
    second = torch.ones(2, 2, dtype=torch.float32) * 2
    third = torch.ones(2, 2, dtype=torch.float32) * 3

    cache.put('first', first)
    cache.put('second', second)
    cache.pin('first')
    cache.put('third', third)

    assert cache.get('first') is not None
    assert cache.get('second') is None
    assert cache.get('third') is not None

    cache.unpin('first')
    cache.put('fourth', torch.ones(2, 2, dtype=torch.float32) * 4)

    assert cache.get('first') is None


class _FakeDLSlimeFuture:

    def wait(self):
        return None


class _FakeDLSlimeEndpoint:

    def __init__(self, name: str):
        self.name = name
        self.connected = []
        self._local_regions = {}
        self._local_keys = {}
        self._remote_regions = {}
        self._mr_info = {}
        self.unregistered = []

    def endpoint_info(self):
        return {'name': self.name}

    def mr_info(self):
        return self._mr_info

    def connect(self, endpoint_info):
        self.connected.append(endpoint_info)

    def register_memory_region(self, name, data_ptr, offset, length):
        handle = len(self._local_regions)
        self._local_keys[name] = handle
        self._local_regions[handle] = {
            'addr': int(data_ptr) + int(offset),
            'length': int(length),
        }
        self._mr_info[name] = {
            'addr': int(data_ptr) + int(offset),
            'handle': handle,
            'length': int(length),
            'rkey': handle,
        }
        return handle

    def unregister_memory_region(self, name):
        self.unregistered.append(name)
        handle = self._local_keys.pop(name, None)
        if handle is not None:
            self._local_regions.pop(handle, None)
        self._mr_info.pop(name, None)

    def register_remote_memory_region(self, name, mr_info):
        handle = 1000 + len(self._remote_regions)
        self._remote_regions[handle] = {
            'addr': int(mr_info['addr']),
            'length': int(mr_info['length']),
        }
        return handle

    def read(self, assignments):
        for local_handle, remote_handle, target_offset, source_offset, length in assignments:
            local = self._local_regions[local_handle]
            remote = self._remote_regions[remote_handle]
            assert int(target_offset) + int(length) <= local['length']
            assert int(source_offset) + int(length) <= remote['length']
            ctypes.memmove(
                local['addr'] + int(target_offset),
                remote['addr'] + int(source_offset),
                int(length),
            )
        return _FakeDLSlimeFuture()

    def shutdown(self):
        return None


class _FakeDLSlimePool:

    def __init__(self):
        self.unregistered = []

    def unregister_memory_region(self, name):
        self.unregistered.append(name)


class _FakeDLSlimeEndpointWithPool:

    def __init__(self):
        self.pool = _FakeDLSlimePool()

    def get_pool(self):
        return self.pool


def _connect_transfer_managers(producer: EncoderTransferManager, consumer: EncoderTransferManager):
    producer_endpoint_info = producer.p2p_initialize(
        EPDInitRequest(
            protocol=MigrationProtocol.RDMA,
            local_engine_id='http://encoder',
            remote_engine_id='http://language',
        ))
    consumer_endpoint_info = consumer.p2p_initialize(
        EPDInitRequest(
            protocol=MigrationProtocol.RDMA,
            local_engine_id='http://language',
            remote_engine_id='http://encoder',
        ))
    producer.p2p_connect('http://language', consumer_endpoint_info)
    consumer.p2p_connect('http://encoder', producer_endpoint_info)


def test_encoder_transfer_manager_round_trip_with_fake_endpoint():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [InputEmbeddings(source, start=1, end=3)],
    }
    producer = EncoderTransferManager('encoder', endpoint=_FakeDLSlimeEndpoint('encoder'), device='cpu')
    consumer = EncoderTransferManager('language', endpoint=_FakeDLSlimeEndpoint('language'), device='cpu')
    _connect_transfer_managers(producer, consumer)

    ref = asyncio.run(
        producer.publish(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_id='epd-test',
        ))
    prompt_input = asyncio.run(consumer.receive(ref))

    assert ref.protocol is MigrationProtocol.RDMA
    assert ref.input_embedding_ranges == [[1, 3]]
    assert ref.shape == [[2, 2]]
    reloaded = EncoderOutputRef.model_validate_json(ref.model_dump_json())
    assert reloaded.transfer_metadata.nbytes == ref.transfer_metadata.nbytes
    assert prompt_input['input_ids'] == [10, 11, 12, 13]
    assert len(prompt_input['input_embeddings']) == 1
    received = prompt_input['input_embeddings'][0]
    assert received.start == 1
    assert received.end == 3
    torch.testing.assert_close(received.embeddings, source)
    assert 'epd_encoder_output_recv' in consumer.endpoint.unregistered
    producer.release_published('epd-test')
    assert producer._published_tensors == {}
    assert 'epd-test' in producer.endpoint.unregistered


def test_encoder_transfer_manager_round_trip_from_cached_entry():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    cache = EncoderCache(max_bytes=1024)
    cache.put('cache-key', source)
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [InputEmbeddings(source, start=1, end=3)],
        'epd_encoder_cache_keys': ['cache-key'],
    }
    producer = EncoderTransferManager('encoder',
                                      endpoint=_FakeDLSlimeEndpoint('encoder'),
                                      device='cpu',
                                      encoder_cache=cache)
    consumer = EncoderTransferManager('language', endpoint=_FakeDLSlimeEndpoint('language'), device='cpu')
    _connect_transfer_managers(producer, consumer)

    ref = asyncio.run(
        producer.publish(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_id='epd-test',
        ))
    prompt_input = asyncio.run(consumer.receive(ref))

    assert ref.transfer_metadata.entries is not None
    assert ref.transfer_metadata.entries[0].cache_key == 'cache-key'
    assert cache.get_entry('cache-key').ref_count == 1
    received = prompt_input['input_embeddings'][0]
    assert received.start == 1
    assert received.end == 3
    torch.testing.assert_close(received.embeddings, source)

    producer.release_published('epd-test')
    assert cache.get_entry('cache-key').ref_count == 0
    cache.clear()
    assert 'epd-cache-cache-key' in producer.endpoint.unregistered


def test_encoder_transfer_manager_materializes_keyed_prompt_embedding_cache():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    cache = EncoderCache(max_bytes=1024)
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [InputEmbeddings(source, start=1, end=3)],
        'epd_encoder_cache_keys': ['cache-key'],
    }
    producer = EncoderTransferManager('encoder',
                                      endpoint=_FakeDLSlimeEndpoint('encoder'),
                                      device='cpu',
                                      encoder_cache=cache)
    consumer = EncoderTransferManager('language', endpoint=_FakeDLSlimeEndpoint('language'), device='cpu')
    _connect_transfer_managers(producer, consumer)

    ref = asyncio.run(
        producer.publish(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_id='epd-test',
        ))
    prompt_input = asyncio.run(consumer.receive(ref))

    assert ref.transfer_metadata.entries is not None
    assert ref.transfer_metadata.entries[0].cache_key == 'cache-key'
    assert cache.get_entry('cache-key').ref_count == 1
    torch.testing.assert_close(prompt_input['input_embeddings'][0].embeddings, source)

    producer.release_published('epd-test')
    cache.clear()
    assert 'epd-cache-cache-key' in producer.endpoint.unregistered


def test_encoder_transfer_manager_rejects_bad_receive_metadata():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [InputEmbeddings(source, start=1, end=3)],
    }
    producer = EncoderTransferManager('encoder', endpoint=_FakeDLSlimeEndpoint('encoder'), device='cpu')
    consumer = EncoderTransferManager('language', endpoint=_FakeDLSlimeEndpoint('language'), device='cpu')
    _connect_transfer_managers(producer, consumer)

    ref = asyncio.run(
        producer.publish(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_id='epd-test',
        ))

    bad_nbytes = ref.model_copy(deep=True)
    bad_nbytes.transfer_metadata.nbytes = int(bad_nbytes.transfer_metadata.nbytes) + 4
    with pytest.raises(ValueError, match='byte size'):
        asyncio.run(consumer.receive(bad_nbytes))

    bad_ranges = ref.model_copy(deep=True)
    bad_ranges.input_embedding_ranges = []
    with pytest.raises(ValueError, match='counts do not match'):
        asyncio.run(consumer.receive(bad_ranges))

    bad_range_len = ref.model_copy(deep=True)
    bad_range_len.input_embedding_ranges = [[1, 4]]
    with pytest.raises(ValueError, match='range length'):
        asyncio.run(consumer.receive(bad_range_len))

    producer.release_published('epd-test')


def test_encoder_transfer_manager_releases_through_endpoint_pool():
    endpoint = _FakeDLSlimeEndpointWithPool()
    manager = EncoderTransferManager('encoder', endpoint=endpoint, device='cpu')
    manager._published_mr_keys['epd-test'] = 'epd-test'

    manager.release_published('epd-test')

    assert endpoint.pool.unregistered == ['epd-test']


class _FakeInputProcessor:

    def preprocess_input(self, input_ids, input_multimodals):
        mm_data = [
            MultiModalData(
                modality=Modality.IMAGE,
                data=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                start=1,
                end=3,
                meta={
                    'grid_thw': torch.tensor([1, 1, 2]),
                    'image_token_id': 99,
                },
            )
        ]
        return PreprocessInputResult(input_ids=input_ids, input_multimodals={'mm_data': mm_data})


class _FakeVisual(nn.Module):

    spatial_merge_size = 1

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.empty(0))
        self.forward_count = 0

    def rot_pos_emb(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def fast_pos_embed_interpolate(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def forward(self, pixel_values, cu_seqlens, rotary_pos_emb, pos_embeds):
        self.forward_count += 1
        return torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32)


class _FakeQwen35InnerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()

    def get_visual_embeddings(self, pixel_values, grid_thw, vis_cu_seqlens, vis_pos_emb, pos_embeds):
        image_embeds = self.visual(pixel_values,
                                   cu_seqlens=vis_cu_seqlens,
                                   rotary_pos_emb=vis_pos_emb,
                                   pos_embeds=pos_embeds)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(image_embeds, split_sizes)


class _FakeQwen35Model(nn.Module):

    compute_encoder_prompt_input = Qwen3_5ForConditionalGeneration.compute_encoder_prompt_input
    _compute_epd_encoder_items_uncached = Qwen3_5ForConditionalGeneration._compute_epd_encoder_items_uncached
    _compute_epd_encoder_items_with_cache = EPDEncoderMixin._compute_epd_encoder_items_with_cache

    def __init__(self, encoder_only=False):
        super().__init__()
        self.encoder_only = encoder_only
        self.language_only = False
        self.config = type('Config', (), {})()
        self.model = _FakeQwen35InnerModel()
        self.input_processor = _FakeInputProcessor()
        self.embed_tokens = nn.Embedding(128, 2)

    def get_input_embeddings(self):
        if self.encoder_only:
            raise RuntimeError('encoder-only fake model does not load token embeddings')
        return self.embed_tokens

    def get_input_processor(self):
        return self.input_processor

    def get_multimodal_mask(self, input_ids, mm_inputs):
        image_token_id = mm_inputs[0].meta['image_token_id']
        return input_ids == image_token_id


class _FakeGraphRunner:

    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def get_input_processor(self):
        return self.model.get_input_processor()


def test_compute_encoder_prompt_input_runs_non_deepstack_visual_path():
    set_epd_encoder_cache(EncoderCache(max_bytes=1024))
    prompt_input = {
        'input_ids': [1, 99, 99, 2],
        'multimodal': [{
            'modality': Modality.IMAGE,
            'pixel_values': torch.ones((2, 1)),
            'image_grid_thw': torch.tensor([1, 1, 2]),
            'offset': (1, 3),
            'image_token_id': 99,
        }],
    }

    computed = _FakeQwen35Model().compute_encoder_prompt_input(prompt_input)

    assert computed['input_ids'] == [1, 99, 99, 2]
    assert 'multimodal' not in computed
    assert computed['input_embedding_ranges'] == [[1, 3]]
    assert computed['epd_encoder_cache_keys'] == [
        build_epd_mm_cache_key(_FakeInputProcessor().preprocess_input(None, None).input_multimodals['mm_data'][0])
    ]
    assert len(computed['input_embeddings']) == 1
    embedding = computed['input_embeddings'][0]
    assert isinstance(embedding, InputEmbeddings)
    assert embedding.start == 1
    assert embedding.end == 3
    np.testing.assert_allclose(embedding.embeddings, np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))


def test_compute_encoder_prompt_input_allows_encoder_only_model_without_token_embeddings():
    prompt_input = {
        'input_ids': [1, 99, 99, 2],
        'multimodal': [{
            'modality': Modality.IMAGE,
            'pixel_values': torch.ones((2, 1)),
            'image_grid_thw': torch.tensor([1, 1, 2]),
            'offset': (1, 3),
            'image_token_id': 99,
        }],
    }

    computed = _FakeQwen35Model(encoder_only=True).compute_encoder_prompt_input(prompt_input)

    assert computed['input_embedding_ranges'] == [[1, 3]]
    np.testing.assert_allclose(computed['input_embeddings'][0].embeddings,
                               np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))


def test_compute_encoder_prompt_input_reuses_cached_embedding():
    set_epd_encoder_cache(EncoderCache(max_bytes=1024))
    model = _FakeQwen35Model()
    prompt_input = {
        'input_ids': [1, 99, 99, 2],
        'multimodal': [{
            'modality': Modality.IMAGE,
            'pixel_values': torch.ones((2, 1)),
            'image_grid_thw': torch.tensor([1, 1, 2]),
            'offset': (1, 3),
            'image_token_id': 99,
        }],
    }

    first = model.compute_encoder_prompt_input(prompt_input)
    second = model.compute_encoder_prompt_input(prompt_input)

    assert model.model.visual.forward_count == 1
    torch.testing.assert_close(first['input_embeddings'][0].embeddings, second['input_embeddings'][0].embeddings)


def test_model_agent_compute_encoder_prompt_input_unwraps_graph_runner_model():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    prompt_input = {
        'input_ids': [1, 99, 99, 2],
        'multimodal': [{
            'modality': Modality.IMAGE,
            'pixel_values': torch.ones((2, 1)),
            'image_grid_thw': torch.tensor([1, 1, 2]),
            'offset': (1, 3),
            'image_token_id': 99,
        }],
    }

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.patched_model = _FakeGraphRunner(_FakeQwen35Model())
    agent.all_context = lambda: contextlib.nullcontext()

    computed = agent.compute_encoder_prompt_input(prompt_input)

    np.testing.assert_allclose(computed['input_embeddings'][0].embeddings,
                               np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))


def test_compute_encoder_prompt_input_for_engine_unwraps_async_engine():
    class _FakeRemoteEngine:

        def __init__(self):
            self.received = None

        async def compute_encoder_prompt_input(self, prompt_input):
            self.received = prompt_input
            return {
                'input_ids': prompt_input['input_ids'],
                'input_embeddings': ['remote-embedding'],
            }

    remote_engine = _FakeRemoteEngine()
    async_engine = type('FakeAsyncEngine', (), {'engine': remote_engine})()
    prompt_input = {'input_ids': [1, 2], 'multimodal': ['raw-mm']}

    computed = asyncio.run(compute_encoder_prompt_input_for_engine(prompt_input, async_engine))

    assert remote_engine.received is prompt_input
    assert computed == {
        'input_ids': [1, 2],
        'input_embeddings': ['remote-embedding'],
    }


def test_mp_executor_compute_method_runs_all_workers_and_returns_rank0():
    from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor

    executor = MPExecutor.__new__(MPExecutor)
    captured = {}

    async def fake_collective_rpc_async(method, args=None, kwargs=None, receiver_mask=0xff, return_mask=0xff):
        captured.update(
            method=method,
            args=args,
            kwargs=kwargs,
            receiver_mask=receiver_mask,
            return_mask=return_mask,
        )
        return ['rank0-computed', None]

    executor.collective_rpc_async = fake_collective_rpc_async
    prompt_input = {'input_ids': [1, 2], 'multimodal': ['raw-mm']}

    computed = asyncio.run(executor.compute_encoder_prompt_input(prompt_input))

    assert computed == 'rank0-computed'
    assert captured['method'] == 'compute_encoder_prompt_input'
    assert captured['args'] == (prompt_input, )
    assert captured['receiver_mask'] == 0xff
    assert captured['return_mask'] == 1


def test_mp_executor_encoder_role_start_skips_output_prefetch():
    from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor

    executor = MPExecutor.__new__(MPExecutor)
    executor.cache_config = type('FakeCacheConfig', (), {'role': EngineRole.Encoder})()
    called = []

    def fake_collective_rpc(method, *args, **kwargs):
        called.append((method, args, kwargs))

    executor.collective_rpc = fake_collective_rpc
    executor.start(None)

    assert called == [('start', (), {})]
    assert executor._prefetch_task is None


def test_chat_completions_endpoint_dispatches_encoder_role_to_epd_helper(monkeypatch):
    from lmdeploy.serve.openai import api_server
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    class _FakeSessionManager:

        def has(self, session_id):
            return False

    fake_async_engine = type('FakeAsyncEngine', (), {
        'model_name': 'm',
        'backend_config': type('FakeBackendConfig', (), {
            'role': EngineRole.Encoder,
            'logprobs_mode': None,
        })(),
        'session_mgr': _FakeSessionManager(),
    })()
    monkeypatch.setattr(api_server.VariableInterface, 'async_engine', fake_async_engine)

    called = {}

    async def fake_encoder_response(request, raw_request):
        called['request'] = request
        called['raw_request'] = raw_request
        return {'object': 'encoder_output_ref'}

    monkeypatch.setattr(api_server, '_create_epd_encoder_response', fake_encoder_response)

    request = ChatCompletionRequest(model='m', messages='hello')
    raw_request = object()

    response = asyncio.run(api_server.chat_completions_v1(request, raw_request))

    assert response == {'object': 'encoder_output_ref'}
    assert called == {
        'request': request,
        'raw_request': raw_request,
    }


def test_epd_control_endpoints_call_transfer_manager(monkeypatch):
    from lmdeploy.serve.openai import api_server

    class _FakeTransferManager:

        def __init__(self):
            self.calls = []

        def p2p_initialize(self, init_request):
            self.calls.append(('init', init_request.remote_engine_id))
            return EncoderTransferEndpointInfo(protocol=init_request.protocol, endpoint_info={'name': 'local'})

        def p2p_connect(self, remote_engine_id, endpoint_info):
            self.calls.append(('connect', remote_engine_id, endpoint_info.endpoint_info))

        def p2p_drop_connect(self, remote_engine_id):
            self.calls.append(('drop', remote_engine_id))

    manager = _FakeTransferManager()
    monkeypatch.setattr(api_server, 'get_encoder_transfer_manager', lambda: manager)

    init_response = asyncio.run(
        api_server.epd_p2p_initialize(
            EPDInitRequest(
                protocol=MigrationProtocol.RDMA,
                local_engine_id='http://encoder',
                remote_engine_id='http://language',
            )))
    assert init_response.status is EPDConnectionStatus.SUCCESS
    assert init_response.encoder_transfer_endpoint_info.endpoint_info == {'name': 'local'}

    conn_response = asyncio.run(
        api_server.epd_p2p_connect(
            EPDConnectionRequest(
                protocol=MigrationProtocol.RDMA,
                remote_engine_id='http://language',
                remote_encoder_transfer_endpoint_info=EncoderTransferEndpointInfo(
                    protocol=MigrationProtocol.RDMA,
                    endpoint_info={'name': 'language'},
                ),
            )))
    assert conn_response.status is EPDConnectionStatus.SUCCESS

    drop_response = asyncio.run(
        api_server.epd_p2p_drop_connect(
            EPDDropConnectionRequest(engine_id='http://encoder', remote_engine_id='http://language')))
    assert drop_response == {'status': 'SUCCESS'}
    assert manager.calls == [
        ('init', 'http://language'),
        ('connect', 'http://language', {'name': 'language'}),
        ('drop', 'http://language'),
    ]


def test_lifespan_initializes_transfer_manager_only_for_epd_nodes(monkeypatch):
    from lmdeploy.serve.openai import api_server

    class _FakeHealthMonitor:

        def __init__(self, async_engine):
            self.async_engine = async_engine

        def start(self):
            return None

        async def stop(self):
            return None

    class _FakeTransferManager:

        def __init__(self, engine_id, rank, **kwargs):
            self.engine_id = engine_id
            created.append((engine_id, rank, kwargs))

        def close(self):
            closed.append(self.engine_id)

    async def _fake_stop_metrics_handler():
        return None

    def _config(role=EngineRole.Hybrid, language_only=False):
        return type('FakeBackendConfig', (), {
            'role': role,
            'language_only': language_only,
            'migration_backend': MigrationBackend.DLSlime,
            'dp_rank': 0,
            'enable_metrics': False,
            'encoder_cache_size_gb': 0.5,
        })()

    async def _run_lifespan(config):
        handler = api_server.create_lifespan_handler(config, object())
        async with handler(None):
            pass

    created = []
    closed = []
    set_managers = []
    monkeypatch.setattr(api_server.VariableInterface, 'proxy_url', 'http://proxy')
    monkeypatch.setattr(api_server.VariableInterface, 'api_server_url', 'http://language')
    monkeypatch.setattr(api_server, 'EngineHealthMonitor', _FakeHealthMonitor)
    monkeypatch.setattr(api_server, 'EncoderTransferManager', _FakeTransferManager)
    monkeypatch.setattr(api_server, 'set_encoder_transfer_manager', set_managers.append)
    monkeypatch.setattr(api_server.metrics_processor, 'stop_metrics_handler', _fake_stop_metrics_handler)

    asyncio.run(_run_lifespan(_config(role=EngineRole.Hybrid, language_only=False)))
    assert created == []
    assert set_managers == []

    asyncio.run(_run_lifespan(_config(role=EngineRole.Hybrid, language_only=True)))
    asyncio.run(_run_lifespan(_config(role=EngineRole.Encoder, language_only=False)))

    assert created == [
        ('http://language', 0, {}),
        ('http://language', 0, {}),
    ]
    assert closed == ['http://language', 'http://language']
    assert len(set_managers) == 4
    assert set_managers[1] is None
    assert set_managers[3] is None


def test_startup_event_advertises_encoder_receiver_only_for_language_only(monkeypatch):
    from lmdeploy.serve.openai import api_server

    class _FakeEngine:
        is_dummy = False

    class _FakeAsyncEngine:
        model_name = 'm'
        engine = _FakeEngine()

        def __init__(self, backend_config):
            self.backend_config = backend_config

        def start_loop(self, loop, use_async_api=True):
            return None

    def _config(language_only=False):
        return type('FakeBackendConfig', (), {
            'role': EngineRole.Hybrid,
            'language_only': language_only,
            'migration_backend': MigrationBackend.DLSlime,
            'adapters': [],
            'logprobs_mode': None,
        })()

    posts = []

    def _fake_post(url, headers, json):
        posts.append(json)
        return type('FakeResponse', (), {'status_code': 200, 'text': ''})()

    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(post=_fake_post))
    monkeypatch.setattr(api_server.VariableInterface, 'proxy_url', 'http://proxy')
    monkeypatch.setattr(api_server.VariableInterface, 'api_server_url', 'http://node')
    monkeypatch.setattr(api_server, 'get_encoder_transfer_manager',
                        lambda: type('FakeManager', (), {'endpoint_info': {'rdma': 'endpoint'}})())

    monkeypatch.setattr(api_server.VariableInterface, 'async_engine', _FakeAsyncEngine(_config(language_only=False)))
    asyncio.run(api_server.startup_event())
    assert posts[-1]['status']['encoder_output_receiver_endpoint_info'] is None

    monkeypatch.setattr(api_server.VariableInterface, 'async_engine', _FakeAsyncEngine(_config(language_only=True)))
    asyncio.run(api_server.startup_event())
    assert posts[-1]['status']['encoder_output_receiver_endpoint_info'] == {'rdma': 'endpoint'}


def test_engine_instance_forwards_input_embeddings_to_add_message():
    class FakeSender:
        sender_id = 0

        def __init__(self):
            self.sent = []

        def send_async(self, request_type, data):
            self.sent.append((request_type, data))
            return object()

        async def async_recv(self, resp, wait_main=True):
            return Response(
                type=ResponseType.FINISH,
                sender_id=self.sender_id,
                event=asyncio.Event(),
                data={'token_ids': np.array([4], dtype=np.int64)},
            )

    async def run_case():
        fake_sender = FakeSender()
        fake_engine = type('FakeEngine', (), {
            'req_manager': type('FakeReqManager', (), {
                'senders': {
                    fake_sender.sender_id: fake_sender,
                },
            })(),
        })()
        instance = EngineInstance.__new__(EngineInstance)
        instance.engine = fake_engine
        instance.req_sender = fake_sender
        instance.max_input_len = 16
        instance._enable_transfer_obj_ref = False

        embeddings = [InputEmbeddings(np.ones((1, 2), dtype=np.float32), start=1, end=2)]
        outputs = [
            output async for output in instance.async_stream_infer(
                session_id=99,
                input_ids=[1, 2],
                gen_config=GenerationConfig(max_new_tokens=1),
                input_embeddings=embeddings,
            )
        ]

        assert len(outputs) == 1
        assert fake_sender.sent[0][0] is RequestType.ADD_SESSION
        assert fake_sender.sent[1][0] is RequestType.ADD_MESSAGE
        assert fake_sender.sent[1][1]['input_embeddings'] is embeddings

    asyncio.run(run_case())
