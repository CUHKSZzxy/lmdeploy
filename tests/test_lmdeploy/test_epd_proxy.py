from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.serve.epd_channel import EPD_BACKEND_ZMQ_IPC
from lmdeploy.serve.proxy.proxy import (
    NodeManager,
    Status,
    _build_epd_encoder_request,
    _build_epd_language_request,
    _has_multimodal_chat_messages,
)


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


def test_build_epd_language_request_preserves_messages_and_injects_encoder_result():
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 0.1,
        'stream': True,
    }
    encoder_result = {
        'token_ids': [1, 2],
        'protocol': MigrationProtocol.TCP.name,
        'backend': 'http_json',
        'remote_engine_id': 'http://encoder',
        'remote_session_id': 3,
        'remote_block_ids': [],
    }

    language_request = _build_epd_language_request(request_dict, encoder_result)

    assert language_request['messages'] == request_dict['messages']
    assert language_request['temperature'] == 0.1
    assert language_request['stream'] is True
    assert language_request['encoder_result']['token_ids'] == [1, 2]
    assert language_request['encoder_result']['protocol'] == 'TCP'


def test_build_epd_encoder_request_uses_language_channel():
    request_dict = {
        'model': 'm',
        'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.jpg'}}]}],
        'stream': True,
    }
    language_status = Status(
        role=EngineRole.Hybrid,
        models=['m'],
        epd_transfer_backend=EPD_BACKEND_ZMQ_IPC,
        epd_channel_address='ipc:///tmp/lmdeploy_epd_test.sock',
    )

    encoder_request = _build_epd_encoder_request(request_dict, language_status)

    assert encoder_request['stream'] is False
    assert encoder_request['encoder_transfer_backend'] == EPD_BACKEND_ZMQ_IPC
    assert encoder_request['epd_channel_address'] == 'ipc:///tmp/lmdeploy_epd_test.sock'
    assert encoder_request['epd_transfer_id'].startswith('epd-')


def test_api_server_does_not_register_dedicated_encoder_chat_endpoint():
    from lmdeploy.serve.openai.api_server import router

    paths = {route.path for route in router.routes}

    assert '/v1/chat/completions' in paths
    assert '/v1/chat/encoder' not in paths
