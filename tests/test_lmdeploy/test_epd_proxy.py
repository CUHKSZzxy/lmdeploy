import asyncio
import importlib
import json

import pytest

from lmdeploy.pytorch.disagg.config import EngineRole, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import EncoderOutputRef, MigrationProtocol
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.proxy.proxy import (
    NodeManager,
    Status,
    _call_epd_encoder_node,
    _dispatch_epd_language_request,
)

proxy_mod = importlib.import_module('lmdeploy.serve.proxy.proxy')


def test_node_manager_tracks_encoder_nodes():
    manager = NodeManager(cache_status=False)
    manager.nodes = {
        'http://encoder': Status(role=EngineRole.Encoder, models=['m']),
        'http://language': Status(role=EngineRole.Hybrid, models=['m']),
    }

    assert manager.encoder_nodes == {'http://encoder': manager.nodes['http://encoder']}
    assert manager.get_node_url('m', EngineRole.Encoder) == 'http://encoder'


def test_call_epd_encoder_node_uses_language_transfer_config():

    class _NodeManager:

        def __init__(self):
            self.request = None

        def pre_call(self, node_url):
            return 1.0

        def post_call(self, node_url, start):
            pass

        async def generate(self, request_dict, node_url, path):
            self.request = request_dict
            return json.dumps({
                'encoder_output_ref': {
                    'token_ids': [1, 2],
                    'input_embedding_ranges': [[0, 2]],
                    'protocol': MigrationProtocol.RDMA.name,
                    'transfer_id': 'epd-test',
                    'remote_engine_id': 'http://encoder',
                    'remote_session_id': 3,
                    'dtype': 'float32',
                    'shape': [[2, 4]],
                    'transfer_metadata': {
                        'endpoint_info': {'name': 'encoder'},
                        'mr_info': {'addr': 0, 'length': 32},
                        'nbytes': 32,
                    },
                }
            })

    endpoint_info = {'io_info': {'data_channel_info': []}}
    language_status = Status(
        role=EngineRole.Hybrid,
        models=['m'],
        encoder_output_receiver_endpoint_info=endpoint_info,
    )
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}}]}],
        'stream': True,
    }
    manager = _NodeManager()

    encoder_output_ref = asyncio.run(
        _call_epd_encoder_node(manager, request_dict, 'http://encoder', 'http://language', language_status))

    assert encoder_output_ref.transfer_id == 'epd-test'
    assert manager.request['stream'] is False
    assert manager.request['encoder_output_receiver_endpoint_info'] == endpoint_info
    assert manager.request['encoder_output_receiver_engine_id'] == 'http://language'
    assert manager.request['epd_transfer_id'].startswith('epd-')


def test_call_epd_encoder_node_balances_node_accounting_on_error():

    class _FailingNodeManager:

        def __init__(self):
            self.calls = []

        def pre_call(self, node_url):
            self.calls.append(('pre', node_url))
            return 1.0

        def post_call(self, node_url, start):
            self.calls.append(('post', node_url, start))

        async def generate(self, request_dict, node_url, path):
            raise RuntimeError('encoder failed')

    endpoint_info = {'io_info': {'data_channel_info': []}}
    language_status = Status(
        role=EngineRole.Hybrid,
        models=['m'],
        encoder_output_receiver_endpoint_info=endpoint_info,
    )
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}}]}],
    }
    manager = _FailingNodeManager()

    with pytest.raises(RuntimeError, match='encoder failed'):
        asyncio.run(_call_epd_encoder_node(manager, request_dict, 'http://encoder', 'http://language',
                                           language_status))

    assert manager.calls == [('pre', 'http://encoder'), ('post', 'http://encoder', 1.0)]


def test_multimodal_hybrid_request_falls_back_when_selected_node_is_not_epd_capable(monkeypatch):

    class _FakeNodeManager:
        serving_strategy = ServingStrategy.Hybrid

        def __init__(self):
            self.nodes = {
                'http://normal': Status(role=EngineRole.Hybrid, models=['m']),
                'http://encoder': Status(role=EngineRole.Encoder, models=['m']),
            }
            self.forwarded = None

        @property
        def encoder_nodes(self):
            return {'http://encoder': self.nodes['http://encoder']}

        async def check_request_model(self, model):
            return None

        def get_node_url(self, model, role=EngineRole.Hybrid):
            if role == EngineRole.Encoder:
                return 'http://encoder'
            return 'http://normal'

        def pre_call(self, node_url):
            return 1.0

        def post_call(self, node_url, start):
            pass

        async def forward_raw_request_generate(self, raw_request, node_url, endpoint):
            self.forwarded = (node_url, endpoint)
            return '{"ok": true}'

    manager = _FakeNodeManager()
    monkeypatch.setattr(proxy_mod, 'node_manager', manager)
    request = ChatCompletionRequest(
        model='m',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe'},
                {'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}},
            ],
        }],
        stream=False,
    )

    response = asyncio.run(proxy_mod.chat_completions_v1(request, raw_request=object()))

    assert response.status_code == 200
    assert response.body == b'{"ok":true}'
    assert manager.forwarded == ('http://normal', '/v1/chat/completions')


def test_dispatch_epd_language_request_injects_and_frees_encoder_output_ref(monkeypatch):

    class _NodeManager:

        def __init__(self):
            self.calls = []
            self.request = None

        def pre_call(self, node_url):
            self.calls.append(('pre', node_url))
            return 1.0

        def post_call(self, node_url, start):
            self.calls.append(('post', node_url, start))

        async def generate(self, request_dict, node_url, path):
            self.request = request_dict
            return '{"ok": true}'

    called = {}

    async def fake_release(encoder_output_ref):
        called['encoder_output_ref'] = encoder_output_ref

    monkeypatch.setattr(proxy_mod, 'free_remote_encoder_cache_ref_async', fake_release)
    manager = _NodeManager()
    request = ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hello'}], stream=False)
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 0.1,
        'stream': False,
    }
    encoder_output_ref = {
        'token_ids': [1, 2],
        'input_embedding_ranges': [[0, 2]],
        'protocol': MigrationProtocol.RDMA.name,
        'transfer_id': 'epd-test',
        'remote_engine_id': 'http://encoder',
        'remote_session_id': 3,
        'dtype': 'float32',
        'shape': [[2, 4]],
        'transfer_metadata': {
            'endpoint_info': {'name': 'encoder'},
            'mr_info': {'addr': 0, 'length': 32},
            'nbytes': 32,
        },
    }

    response = asyncio.run(
        _dispatch_epd_language_request(manager, request, request_dict, 'http://language',
                                       EncoderOutputRef.model_validate(encoder_output_ref)))

    assert response.status_code == 200
    assert response.body == b'{"ok":true}'
    assert manager.request['temperature'] == 0.1
    assert manager.request['encoder_output_ref']['transfer_id'] == 'epd-test'
    assert manager.calls == [('pre', 'http://language'), ('post', 'http://language', 1.0)]
    assert called['encoder_output_ref'].transfer_id == 'epd-test'
