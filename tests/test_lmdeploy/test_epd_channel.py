import asyncio
import os
import subprocess
import sys

import numpy as np
import pytest

from lmdeploy.serve.epd_channel import (
    EncoderTransferEmbedding,
    EncoderTransferPayload,
    _frames_to_payload,
    _payload_to_frames,
    recv_epd_payload,
    send_epd_payload,
    close_epd_senders,
    start_epd_receiver,
    stop_epd_receiver,
)


def test_epd_channel_payload_serialization_round_trip():
    payload = EncoderTransferPayload(
        transfer_id='epd-test',
        token_ids=[1, 2, 3],
        embeddings=[
            EncoderTransferEmbedding(
                data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                start=1,
                end=3,
                dtype='float32',
            )
        ],
    )

    received = _frames_to_payload(_payload_to_frames(payload))

    assert received.transfer_id == 'epd-test'
    assert received.token_ids == [1, 2, 3]
    assert received.input_embedding_ranges == [[1, 3]]
    np.testing.assert_allclose(received.embeddings[0].data, payload.embeddings[0].data)


def _zmq_bind_preflight() -> tuple[bool, str]:
    script = """
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
try:
    sock.bind('tcp://127.0.0.1:*')
finally:
    sock.close(0)
    ctx.term()
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def test_zmq_ipc_epd_channel_round_trip():
    if os.getenv('LMDEPLOY_RUN_ZMQ_TESTS') != '1':
        pytest.skip('set LMDEPLOY_RUN_ZMQ_TESTS=1 to run live ZMQ channel test')

    ok, output = _zmq_bind_preflight()
    if not ok:
        pytest.fail(f'ZMQ bind preflight failed: {output}')

    async def run_case():
        address = 'tcp://127.0.0.1:*'
        receiver = await start_epd_receiver(address)
        try:
            address = receiver.address
            payload = EncoderTransferPayload(
                transfer_id='epd-test',
                token_ids=[1, 2, 3],
                embeddings=[
                    EncoderTransferEmbedding(
                        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                        start=1,
                        end=3,
                        dtype='float32',
                    )
                ],
            )
            await send_epd_payload(address, payload)
            received = await recv_epd_payload('epd-test', timeout=5.0)

            assert received.token_ids == [1, 2, 3]
            assert received.input_embedding_ranges == [[1, 3]]
            np.testing.assert_allclose(received.embeddings[0].data, payload.embeddings[0].data)
        finally:
            await stop_epd_receiver()
            close_epd_senders()

    asyncio.run(run_case())
