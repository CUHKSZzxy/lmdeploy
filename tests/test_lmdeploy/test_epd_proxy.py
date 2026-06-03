import asyncio
import importlib

import pytest

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.serve.proxy.proxy import (
    NodeManager,
    Status,
    _build_epd_encoder_request,
    _build_epd_language_request,
    _call_epd_encoder_node,
    _release_epd_encoder_output_ref,
    _has_multimodal_chat_messages,
)

proxy_mod = importlib.import_module('lmdeploy.serve.proxy.proxy')


def test_engine_role_has_encoder():
    assert EngineRole.Encoder.name == 'Encoder'


def test_node_manager_tracks_encoder_nodes():
    manager = NodeManager(cache_status=False)
    manager.nodes = {
        'http://encoder': Status(role=EngineRole.Encoder, models=['m']),
        'http://language': Status(role=EngineRole.Hybrid, models=['m']),
    }

    assert manager.encoder_nodes == {'http://encoder': manager.nodes['http://encoder']}
    assert manager.get_node_url('m', EngineRole.Encoder) == 'http://encoder'


def test_multimodal_chat_detection():
    assert _has_multimodal_chat_messages([
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe'},
                {'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}},
            ],
        }
    ])
    assert not _has_multimodal_chat_messages([
        {
            'role': 'user',
            'content': 'hello',
        }
    ])


def test_build_epd_language_request_preserves_messages_and_injects_encoder_output_ref():
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 0.1,
        'stream': True,
    }
    encoder_output_ref = {
        'token_ids': [1, 2],
        'protocol': MigrationProtocol.RDMA.name,
        'transfer_id': 'epd-test',
        'remote_engine_id': 'http://encoder',
        'remote_session_id': 3,
        'remote_block_ids': [],
    }

    language_request = _build_epd_language_request(request_dict, encoder_output_ref)

    assert language_request['messages'] == request_dict['messages']
    assert language_request['temperature'] == 0.1
    assert language_request['stream'] is True
    assert language_request['encoder_output_ref']['token_ids'] == [1, 2]
    assert language_request['encoder_output_ref']['protocol'] == 'RDMA'


def test_build_epd_encoder_request_uses_language_rdma_endpoint_info():
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}}]}],
        'stream': True,
    }
    endpoint_info = {'io_info': {'data_channel_info': []}}
    language_status = Status(
        role=EngineRole.Hybrid,
        models=['m'],
        encoder_output_receiver_endpoint_info=endpoint_info,
    )

    encoder_request = _build_epd_encoder_request(request_dict, 'http://language', language_status)

    assert encoder_request['stream'] is False
    assert encoder_request['encoder_output_receiver_endpoint_info'] == endpoint_info
    assert encoder_request['encoder_output_receiver_engine_id'] == 'http://language'
    assert encoder_request['epd_transfer_id'].startswith('epd-')


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


def test_release_epd_encoder_output_ref_uses_connector_cleanup(monkeypatch):
    called = {}

    async def fake_release(encoder_output_ref):
        called['encoder_output_ref'] = encoder_output_ref

    monkeypatch.setattr(proxy_mod, 'release_remote_encoder_output_async', fake_release)
    encoder_output_ref = {
        'token_ids': [1, 2],
        'protocol': MigrationProtocol.RDMA.name,
        'transfer_id': 'epd-test',
        'remote_engine_id': 'http://encoder',
        'remote_session_id': 3,
        'remote_block_ids': [],
    }

    asyncio.run(_release_epd_encoder_output_ref(encoder_output_ref))

    assert called['encoder_output_ref'].transfer_id == 'epd-test'
