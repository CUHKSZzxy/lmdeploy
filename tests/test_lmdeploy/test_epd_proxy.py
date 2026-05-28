from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.epd.connector import EPD_BACKEND_DLSLIME
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
        epd_transfer_backend=EPD_BACKEND_DLSLIME,
        encoder_output_receiver_endpoint_info=endpoint_info,
    )

    encoder_request = _build_epd_encoder_request(request_dict, 'http://language', language_status)

    assert encoder_request['stream'] is False
    assert encoder_request['encoder_transfer_backend'] == EPD_BACKEND_DLSLIME
    assert encoder_request['encoder_output_receiver_endpoint_info'] == endpoint_info
    assert encoder_request['encoder_output_receiver_engine_id'] == 'http://language'
    assert encoder_request['epd_transfer_id'].startswith('epd-')
