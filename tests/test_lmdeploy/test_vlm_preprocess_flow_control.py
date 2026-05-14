import asyncio

import pytest
import torch

from lmdeploy.messages import VisionConfig
from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient
from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLInputProcessor
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.constants import Modality


def test_vision_config_rejects_invalid_preprocess_workers():
    with pytest.raises(ValueError, match='max_preprocess_workers'):
        VisionConfig(max_preprocess_workers=0)


def test_text_prompt_does_not_acquire_multimodal_permit():

    async def run_case():
        processor = MultimodalProcessor(tokenizer=None, chat_template=None)

        release = await processor.acquire_multimodal_request('hello')

        assert release is None

    asyncio.run(run_case())


def test_multimodal_request_permit_limits_active_preprocessing():

    async def run_case():

        class FakeEncoder:
            vision_config = VisionConfig(max_preprocess_workers=1)

        processor = MultimodalProcessor(tokenizer=None, chat_template=None, vl_encoder=FakeEncoder(), backend='pytorch')
        prompt = [{'role': 'user', 'content': [{'type': 'image_data', 'image_data': {'data': object()}}]}]
        active = 0
        max_active = 0

        async def worker():
            nonlocal active, max_active
            release = await processor.acquire_multimodal_request(prompt)
            try:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
            finally:
                release()

        try:
            await asyncio.gather(*(worker() for _ in range(4)))
            assert max_active == 1
        finally:
            processor.close()

    asyncio.run(run_case())


def test_multimodal_parse_uses_supplied_executor(monkeypatch):

    async def run_case():
        loop = asyncio.get_event_loop()
        executor = object()
        seen_executors = []

        def fake_run_in_executor(seen_executor, func, *args):
            seen_executors.append(seen_executor)
            future = loop.create_future()
            func(*args)
            future.set_result(None)
            return future

        monkeypatch.setattr(loop, 'run_in_executor', fake_run_in_executor)
        messages = [{'role': 'user', 'content': 'hello'} for _ in range(3)]

        await MultimodalProcessor.async_parse_multimodal_item(messages, executor=executor)

        assert seen_executors == [executor, executor, executor]

    asyncio.run(run_case())


def test_qwen3_vl_input_processor_reuses_cached_mm_payload():
    processor = Qwen3VLInputProcessor(config=None, dtype=torch.float32)
    image_grid_thw = torch.tensor([1, 2, 2])
    first_pixels = torch.ones((4, 3), dtype=torch.float32)
    second_pixels = torch.zeros((4, 3), dtype=torch.float32)

    first = processor._make_image_mm_data(
        dict(modality=Modality.IMAGE,
             pixel_values=first_pixels,
             image_grid_thw=image_grid_thw,
             offset=(10, 14),
             image_token_id=99,
             cache_key='same-image'))
    second = processor._make_image_mm_data(
        dict(modality=Modality.IMAGE,
             pixel_values=second_pixels,
             image_grid_thw=image_grid_thw,
             offset=(20, 24),
             image_token_id=99,
             cache_key='same-image'))

    assert second.data is first.data
    assert second.start == 20
    assert second.end == 24


def test_zmq_stream_notify_runs_after_stream_creation(monkeypatch):

    async def run_case():
        client = AsyncRPCClient.__new__(AsyncRPCClient)
        client.pending = {}
        calls = []
        notified = False
        wait_forever = asyncio.Event()

        async def fake_async_call_impl(method, streaming, *args, **kwargs):
            calls.append((method, streaming, kwargs))
            return 7

        async def fake_async_call(method, *args, **kwargs):
            await wait_forever.wait()

        def notify():
            nonlocal notified
            notified = True

        monkeypatch.setattr(client, '_async_call_impl', fake_async_call_impl)
        monkeypatch.setattr(client, 'async_call', fake_async_call)

        task = asyncio.create_task(
            anext(
                client.async_stream_call('instance_async_stream_infer',
                                         asyncio.Event(),
                                         input_ids=[1, 2, 3],
                                         notify_add_msg_func=notify)))
        await asyncio.sleep(0)

        assert calls == [('instance_async_stream_infer', True, {'input_ids': [1, 2, 3]})]
        assert notified

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())
