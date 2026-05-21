import asyncio

import numpy as np
import pytest
import torch
from torch import nn

from lmdeploy.messages import GenerationConfig, ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import EncoderCacheRef, EncoderInlineEmbedding, MigrationProtocol
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.input_process import PreprocessInputResult
from lmdeploy.pytorch.engine.request import RequestType, Response
from lmdeploy.pytorch.messages import InputEmbeddings
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.serve.epd import (
    encoder_cache_ref_to_prompt_input,
    materialize_encoder_prompt_input,
    prompt_input_to_encoder_cache_ref,
)
from lmdeploy.vl.constants import Modality


def test_encoder_cache_ref_round_trip():
    ref = EncoderCacheRef(
        token_ids=[1, 2, 3],
        mm_mask=[0, 1, 0],
        input_embeddings=[
            EncoderInlineEmbedding(
                data=[[0.5, 1.5], [2.5, 3.5]],
                start=1,
                end=3,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='inline',
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


def test_encoder_cache_ref_converts_inline_embeddings_to_prompt_input():
    ref = EncoderCacheRef(
        token_ids=[101, 102, 103, 104],
        input_embeddings=[
            EncoderInlineEmbedding(
                data=[[1.0, 2.0], [3.0, 4.0]],
                start=1,
                end=3,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='inline',
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


def test_prompt_input_converts_to_inline_encoder_cache_ref():
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
    assert ref.backend == 'inline'
    assert ref.remote_engine_id == 'http://encoder'
    assert ref.remote_session_id == 5
    assert ref.dtype == 'float32'
    assert ref.shape == [[2, 2]]
    assert ref.input_embedding_ranges == [[1, 3]]
    assert ref.input_embeddings[0].start == 1
    assert ref.input_embeddings[0].end == 3
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

    def rot_pos_emb(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def fast_pos_embed_interpolate(self, grid_thw):
        return torch.zeros((int(grid_thw.prod().item()), 2), dtype=torch.float32)

    def forward(self, pixel_values, cu_seqlens, rotary_pos_emb, pos_embeds):
        return torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32)


class _FakeQwen35Model(nn.Module):

    def __init__(self, deepstack_visual_indexes=None):
        super().__init__()
        vision_config = type('VisionConfig', (), {
            'deepstack_visual_indexes': deepstack_visual_indexes or [],
        })()
        self.config = type('Config', (), {'vision_config': vision_config})()
        self.model = type('InnerModel', (), {'visual': _FakeVisual()})()
        self.input_processor = _FakeInputProcessor()
        self.embed_tokens = nn.Embedding(128, 2)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_input_processor(self):
        return self.input_processor

    def get_multimodal_mask(self, input_ids, mm_inputs):
        image_token_id = mm_inputs[0].meta['image_token_id']
        return input_ids == image_token_id


def test_materialize_encoder_prompt_input_runs_non_deepstack_visual_path():
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

    materialized = materialize_encoder_prompt_input(prompt_input, _FakeQwen35Model())

    assert materialized['input_ids'] == [1, 99, 99, 2]
    assert 'multimodal' not in materialized
    assert materialized['input_embedding_ranges'] == [[1, 3]]
    assert len(materialized['input_embeddings']) == 1
    embedding = materialized['input_embeddings'][0]
    assert isinstance(embedding, InputEmbeddings)
    assert embedding.start == 1
    assert embedding.end == 3
    np.testing.assert_allclose(embedding.embeddings, np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))


def test_materialize_encoder_prompt_input_rejects_deepstack_visual_model():
    prompt_input = {'input_ids': [1, 99], 'multimodal': [{'modality': Modality.IMAGE}]}

    with pytest.raises(ValueError, match='DeepStack'):
        materialize_encoder_prompt_input(prompt_input, _FakeQwen35Model(deepstack_visual_indexes=[5]))


def test_encoder_cache_ref_rejects_mismatched_inline_embedding_range():
    ref = EncoderCacheRef(
        token_ids=[101, 102, 103],
        input_embeddings=[
            EncoderInlineEmbedding(
                data=[[1.0, 2.0], [3.0, 4.0]],
                start=1,
                end=2,
                dtype='float32',
            )
        ],
        protocol=MigrationProtocol.TCP,
        backend='inline',
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
        backend='inline',
        remote_engine_id='encoder-0',
        remote_session_id=1,
        remote_block_ids=[],
    )

    config = GenerationConfig(encoder_result=ref)

    assert config.encoder_result is ref


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
