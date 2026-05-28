import asyncio
import contextlib
import ctypes

import numpy as np
import pytest
import torch
from torch import nn

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, EncoderHttpJsonEmbedding, MigrationProtocol
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.input_process import PreprocessInputResult
from lmdeploy.pytorch.engine.request import RequestType, Response
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from lmdeploy.pytorch.models.utils.model import build_language_model
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.disagg.epd.connector import (
    EncoderTransferConfig,
    EPD_BACKEND_DLSLIME,
    EPD_BACKEND_HTTP_JSON,
    build_encoder_transfer_config,
    encoder_cache_ref_to_prompt_input,
    get_encoder_transfer_connector,
    prompt_input_to_encoder_cache_ref,
    publish_encoder_output,
)
from lmdeploy.pytorch.disagg.epd.dlslime import (
    DLSlimeEncoderTransferManager,
    set_dlslime_encoder_transfer_manager,
)
from lmdeploy.pytorch.disagg.epd.engine import (
    compute_encoder_prompt_input_for_engine,
)
from lmdeploy.vl.constants import Modality


def test_encoder_cache_ref_round_trip():
    ref = EncoderCacheRef(
        token_ids=[1, 2, 3],
        mm_mask=[0, 1, 0],
        input_embeddings=[
            EncoderHttpJsonEmbedding(
                data=[[0.5, 1.5], [2.5, 3.5]],
                start=1,
                end=3,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='http_json',
        remote_engine_id='encoder-0',
        remote_session_id=7,
        remote_block_ids=[11, 12],
        dtype='float32',
        shape=[2, 2],
        modality='image',
    )

    dumped = ref.model_dump(mode='json')
    loaded = EncoderCacheRef.model_validate(dumped)

    assert loaded.token_ids == [1, 2, 3]
    assert loaded.protocol is MigrationProtocol.TCP
    assert loaded.input_embeddings[0].start == 1
    assert loaded.input_embeddings[0].end == 3


def test_encoder_cache_ref_converts_http_json_embeddings_to_prompt_input():
    ref = EncoderCacheRef(
        token_ids=[101, 102, 103, 104],
        input_embeddings=[
            EncoderHttpJsonEmbedding(
                data=[[1.0, 2.0], [3.0, 4.0]],
                start=1,
                end=3,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='http_json',
        remote_engine_id='encoder-0',
        remote_session_id=9,
        remote_block_ids=[],
    )

    prompt_input = encoder_cache_ref_to_prompt_input(ref)

    assert prompt_input['prompt'] is None
    assert prompt_input['input_ids'] == [101, 102, 103, 104]
    assert len(prompt_input['input_embeddings']) == 1
    embedding = prompt_input['input_embeddings'][0]
    assert isinstance(embedding, InputEmbeddings)
    assert embedding.start == 1
    assert embedding.end == 3
    assert embedding.embeddings.dtype == np.float32
    np.testing.assert_allclose(embedding.embeddings, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_prompt_input_converts_to_http_json_encoder_cache_ref():
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        'input_embedding_ranges': [[1, 3]],
    }

    ref = prompt_input_to_encoder_cache_ref(
        prompt_input,
        remote_engine_id='http://encoder',
        remote_session_id=5,
        protocol=MigrationProtocol.TCP,
    )

    assert ref.token_ids == [10, 11, 12, 13]
    assert ref.backend == 'http_json'
    assert ref.remote_engine_id == 'http://encoder'
    assert ref.remote_session_id == 5
    assert ref.dtype == 'float32'
    assert ref.shape == [[2, 2]]
    assert ref.input_embedding_ranges == [[1, 3]]
    assert ref.input_embeddings[0].start == 1
    assert ref.input_embeddings[0].end == 3
    assert ref.input_embeddings[0].data == [[1.0, 2.0], [3.0, 4.0]]


def test_engine_config_rejects_language_only_with_encoder_only():
    with pytest.raises(ValueError, match='language_only and encoder_only'):
        PytorchEngineConfig(language_only=True, encoder_only=True)


def test_build_language_model_skips_module_in_encoder_only_context():

    class _TinyLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    with build_model_context(BuildModelContext(encoder_only=True)):
        model = build_language_model(_TinyLanguageModel)

    assert isinstance(model, nn.Identity)
    assert model._is_dummy_mod


def test_dlslime_encoder_transfer_config_generates_transfer_id_and_requires_manager():
    endpoint_info = {'io_info': {'data_channel_info': []}}
    config = build_encoder_transfer_config(EPD_BACKEND_DLSLIME,
                                           receiver_endpoint_info=endpoint_info,
                                           receiver_engine_id='http://language')

    assert config.backend == EPD_BACKEND_DLSLIME
    assert config.transfer_id.startswith('epd-')
    assert config.receiver_endpoint_info == endpoint_info
    assert config.to_request_fields() == {
        'encoder_transfer_backend': EPD_BACKEND_DLSLIME,
        'epd_transfer_id': config.transfer_id,
        'encoder_output_receiver_endpoint_info': endpoint_info,
        'encoder_output_receiver_engine_id': 'http://language',
    }

    connector = get_encoder_transfer_connector(EPD_BACKEND_DLSLIME)
    set_dlslime_encoder_transfer_manager(None)
    with pytest.raises(ValueError, match='not initialized'):
        asyncio.run(
            connector.publish(
                {},
                remote_engine_id='http://encoder',
                remote_session_id=5,
                transfer_config=config,
            ))


class _FakeDLSlimeFuture:

    def wait(self):
        return None


class _FakeDLSlimeEndpoint:

    def __init__(self, name: str):
        self.name = name
        self.connected = []
        self._local_regions = {}
        self._remote_regions = {}
        self._mr_info = {}

    def endpoint_info(self):
        return {'name': self.name}

    def mr_info(self):
        return self._mr_info

    def connect(self, endpoint_info):
        self.connected.append(endpoint_info)

    def register_memory_region(self, name, data_ptr, offset, length):
        handle = len(self._local_regions)
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


def test_dlslime_encoder_transfer_manager_round_trip_with_fake_endpoint():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [InputEmbeddings(source, start=1, end=3)],
    }
    producer = DLSlimeEncoderTransferManager('encoder', endpoint=_FakeDLSlimeEndpoint('encoder'), device='cpu')
    consumer = DLSlimeEncoderTransferManager('language', endpoint=_FakeDLSlimeEndpoint('language'), device='cpu')

    ref = asyncio.run(
        producer.publish(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_id='epd-test',
            receiver_endpoint_info=consumer.endpoint_info,
            receiver_engine_id='http://language',
        ))
    prompt_input = asyncio.run(consumer.receive(ref))

    assert ref.backend == EPD_BACKEND_DLSLIME
    assert ref.protocol is MigrationProtocol.RDMA
    assert ref.input_embeddings is None
    assert ref.input_embedding_ranges == [[1, 3]]
    assert ref.shape == [[2, 2]]
    assert prompt_input['input_ids'] == [10, 11, 12, 13]
    assert len(prompt_input['input_embeddings']) == 1
    received = prompt_input['input_embeddings'][0]
    assert received.start == 1
    assert received.end == 3
    torch.testing.assert_close(received.embeddings, source)


def test_http_json_connector_publishes_encoder_cache_ref():
    prompt_input = {
        'input_ids': [10, 11, 12, 13],
        'input_embeddings': [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        'input_embedding_ranges': [[1, 3]],
    }

    ref = asyncio.run(
        publish_encoder_output(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            transfer_config=EncoderTransferConfig(backend='http_json'),
        ))

    assert ref.backend == 'http_json'
    assert ref.remote_engine_id == 'http://encoder'
    assert ref.input_embeddings[0].start == 1
    assert ref.input_embeddings[0].data == [[1.0, 2.0], [3.0, 4.0]]


def test_prompt_input_rejects_pixel_value_multimodal_without_embeddings():
    prompt_input = {
        'input_ids': [10, 11],
        'multimodal': [{'pixel_values': torch.ones((1, 2)), 'offset': 1}],
    }

    with pytest.raises(ValueError, match='precomputed input_embeddings'):
        prompt_input_to_encoder_cache_ref(
            prompt_input,
            remote_engine_id='http://encoder',
            remote_session_id=5,
            protocol=MigrationProtocol.TCP,
        )


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

    def rot_pos_emb(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def fast_pos_embed_interpolate(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def forward(self, pixel_values, cu_seqlens, rotary_pos_emb, pos_embeds):
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

    _prepare_visual_forward_inputs = Qwen3_5ForConditionalGeneration._prepare_visual_forward_inputs
    _make_encoder_input_embeddings = Qwen3_5ForConditionalGeneration._make_encoder_input_embeddings
    compute_encoder_prompt_input = Qwen3_5ForConditionalGeneration.compute_encoder_prompt_input

    def __init__(self, deepstack_visual_indexes=None, encoder_only=False):
        super().__init__()
        self.encoder_only = encoder_only
        self.language_only = False
        vision_config = type('VisionConfig', (), {
            'deepstack_visual_indexes': deepstack_visual_indexes or [],
        })()
        self.config = type('Config', (), {'vision_config': vision_config})()
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


def test_compute_encoder_prompt_input_for_engine_uses_mp_worker_compute_method():
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


def test_pytorch_engine_compute_method_delegates_to_executor():
    from lmdeploy.pytorch.engine.engine import Engine

    class _FakeExecutor:

        def __init__(self):
            self.received = None

        async def compute_encoder_prompt_input(self, prompt_input):
            self.received = prompt_input
            return {
                'input_ids': prompt_input['input_ids'],
                'input_embeddings': ['executor-embedding'],
            }

    executor = _FakeExecutor()
    engine = Engine.__new__(Engine)
    engine.executor = executor
    prompt_input = {'input_ids': [1, 2], 'multimodal': ['raw-mm']}

    computed = asyncio.run(engine.compute_encoder_prompt_input(prompt_input))

    assert executor.received is prompt_input
    assert computed == {
        'input_ids': [1, 2],
        'input_embeddings': ['executor-embedding'],
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
    from lmdeploy.pytorch.disagg.config import EngineRole
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


def test_compute_encoder_prompt_input_rejects_deepstack_visual_model():
    prompt_input = {'input_ids': [1, 99], 'multimodal': [{'modality': Modality.IMAGE}]}

    with pytest.raises(ValueError, match='DeepStack'):
        _FakeQwen35Model(deepstack_visual_indexes=[5]).compute_encoder_prompt_input(prompt_input)


def test_encoder_cache_ref_rejects_mismatched_http_json_embedding_range():
    ref = EncoderCacheRef(
        token_ids=[101, 102, 103],
        input_embeddings=[
            EncoderHttpJsonEmbedding(
                data=[[1.0, 2.0], [3.0, 4.0]],
                start=1,
                end=2,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='http_json',
        remote_engine_id='encoder-0',
        remote_session_id=9,
        remote_block_ids=[],
    )

    with pytest.raises(ValueError, match='does not match embedding rows'):
        encoder_cache_ref_to_prompt_input(ref)


def test_generation_config_carries_encoder_result():
    ref = EncoderCacheRef(
        token_ids=[1],
        protocol=MigrationProtocol.TCP,
        backend='http_json',
        remote_engine_id='encoder-0',
        remote_session_id=1,
        remote_block_ids=[],
    )

    config = GenerationConfig(encoder_result=ref)

    assert config.encoder_result is ref


def test_chat_completions_endpoint_dispatches_encoder_role_to_epd_helper(monkeypatch):
    from lmdeploy.pytorch.disagg.config import EngineRole
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
        return {'object': 'encoder_result'}

    monkeypatch.setattr(api_server, '_create_epd_encoder_response', fake_encoder_response)

    request = ChatCompletionRequest(model='m', messages='hello')
    raw_request = object()

    response = asyncio.run(api_server.chat_completions_v1(request, raw_request))

    assert response == {'object': 'encoder_result'}
    assert called == {
        'request': request,
        'raw_request': raw_request,
    }


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
