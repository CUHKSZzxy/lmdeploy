# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import PytorchEngineConfig, QuantPolicy
from lmdeploy.pytorch.backends.cp_utils import (
    compact_dcp_local_indices,
    fill_dcp_local_seq_lens,
    get_dcp_local_cu_seqlens,
    get_dcp_local_indices,
    get_dcp_local_seq_lens,
)
from lmdeploy.pytorch.config import CacheConfig, DistConfig, MiscConfig
from lmdeploy.pytorch.disagg.config import EngineRole


@pytest.mark.parametrize('world_size', [2, 4])
def test_dcp_local_lengths_match_interleaved_slices(world_size):
    lengths = torch.arange(0, 2 * 64 * world_size + 2, dtype=torch.int32)
    for rank in range(world_size):
        dcp_world_rank = world_size, rank
        local = get_dcp_local_seq_lens(lengths, dcp_world_rank)
        expected = torch.tensor(
            [len(range(rank, int(length), world_size)) for length in lengths],
            dtype=torch.int32)
        assert torch.equal(local, expected)

        local, cu_local = get_dcp_local_cu_seqlens(lengths, dcp_world_rank)
        assert torch.equal(cu_local[1:] - cu_local[:-1], local)
        assert cu_local.dtype == torch.int32


@pytest.mark.parametrize('world_size', [2, 4])
def test_dcp_global_indices_map_to_their_owner(world_size):
    global_indices = torch.tensor([-1, 0, 1, 2, 3, 63, 64, 65, 255],
                                  dtype=torch.int32)
    for rank in range(world_size):
        dcp_world_rank = world_size, rank
        local = get_dcp_local_indices(global_indices, dcp_world_rank)
        owned = (global_indices >= 0) & (global_indices % world_size == rank)
        assert torch.equal(local[owned], global_indices[owned] // world_size)
        assert torch.all(local[~owned] == -1)


def test_fill_dcp_lengths_reuses_caller_owned_buffers():
    global_lens = torch.tensor([0, 1, 3, 8], dtype=torch.int32)
    local_lens = torch.empty_like(global_lens)
    local_ptr = local_lens.data_ptr()

    fill_dcp_local_seq_lens(global_lens, local_lens, (2, 1))
    assert local_lens.data_ptr() == local_ptr
    assert local_lens.tolist() == [0, 0, 1, 4]


def test_dcp_local_winners_are_compacted_with_valid_counts():
    global_indices = torch.tensor(
        [[1, 8, 3, -1, 6, 4], [5, 7, -1, -1, -1, -1]], dtype=torch.int32)
    local, counts = compact_dcp_local_indices(global_indices, (2, 0))

    assert counts.tolist() == [3, 0]
    assert local.tolist() == [[4, 3, 2, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]


def test_tensor_collectives_resolve_dcp_group(monkeypatch):
    from lmdeploy.pytorch import distributed

    dcp_group = object()
    resolved_groups = []
    collective_groups = []

    def get_group(group_type, device):
        resolved_groups.append((group_type, device))
        return dcp_group

    def all_gather_into_tensor(output,
                               input_tensor,
                               group=None,
                               async_op=False):
        collective_groups.append(group)

    def reduce_scatter_tensor(output,
                              input_tensor,
                              op=None,
                              group=None,
                              async_op=False):
        collective_groups.append(group)

    monkeypatch.setattr(distributed, 'get_group', get_group)
    monkeypatch.setattr(distributed.dist, 'all_gather_into_tensor',
                        all_gather_into_tensor)
    monkeypatch.setattr(distributed.dist, 'reduce_scatter_tensor',
                        reduce_scatter_tensor)

    tensor = torch.empty(1)
    distributed.all_gather_into_tensor(tensor, tensor, group='dcp')
    distributed.reduce_scatter_tensor(tensor, tensor, group='dcp')

    assert resolved_groups == [('dcp', 'gpu'), ('dcp', 'gpu')]
    assert collective_groups == [dcp_group, dcp_group]


def test_dcp_candidate_merge_is_exact_and_uses_global_position_ties(
        monkeypatch):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends.cuda.nsa import TritonNSAIndexFP8

    impl = object.__new__(TritonNSAIndexFP8)
    impl.topk = 3
    impl.fill = -1
    impl.dcp_world_size = 2
    impl.dcp_rank = 0

    remote_scores = torch.tensor([[10.0, 7.0, 1.0]])
    remote_positions = torch.tensor([[1, 3, 5]], dtype=torch.int32)
    remote_packed = torch.empty(1, 3, 2, dtype=torch.float32)
    remote_packed[..., 0].copy_(remote_scores)
    remote_packed.view(torch.int32)[..., 1].copy_(remote_positions)
    collective_calls = 0

    def fake_all_gather(output, input_tensor, group='tp', async_op=False):
        nonlocal collective_calls
        collective_calls += 1
        assert group == 'dcp'
        output[:1].copy_(input_tensor)
        output[1:].copy_(remote_packed)

    monkeypatch.setattr(distributed, 'all_gather_into_tensor', fake_all_gather)
    local_scores = torch.tensor([[5.0, 9.0, 7.0]])
    local_indices = torch.tensor([[1, 2, 0]], dtype=torch.int32)

    selected = impl._merge_dcp_topk(local_scores, local_indices)

    # Global winners are score-10 position 1, score-9 position 2, then the
    # lower global position among the equal score-7 candidates (3 before 4).
    assert selected.tolist() == [[1, 2, 3]]
    assert collective_calls == 1


def test_dcp_configuration():
    engine_config = PytorchEngineConfig(tp=8, dcp=4)
    dist_config = DistConfig.from_engine_config(engine_config)
    cache_config = CacheConfig(max_batches=4,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=100,
                               dcp=4)

    assert dist_config.dcp == 4
    assert cache_config.block_size == 64

    with pytest.raises(AssertionError, match='tp must be divisible'):
        PytorchEngineConfig(tp=4, dcp=3)
    with pytest.raises(AssertionError, match='must be divisible'):
        DistConfig(tp=6, dcp=4)


def test_dcp_block_allocation_uses_virtual_block_size():
    from lmdeploy.pytorch.engine.engine import _build_seq_meta
    from lmdeploy.pytorch.paging.block_manager import build_block_manager
    from lmdeploy.pytorch.strategies.ar.sequence import SchedulerSequenceDefault

    cache_config = CacheConfig(max_batches=4,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               dcp=4)
    seq_meta = _build_seq_meta(SimpleNamespace(use_mrope=False),
                               cache_config,
                               seq_strategy=None,
                               sampling_strategy=None)
    session = SimpleNamespace(seq_meta=seq_meta)
    sequence = SchedulerSequenceDefault(seq_id=0, session=session)
    sequence._num_token_ids = 256
    block_manager = build_block_manager(cache_config)

    assert seq_meta.block_size == 64
    assert block_manager.num_required_blocks(sequence) == 1
    sequence._num_token_ids += 1
    assert block_manager.num_required_blocks(sequence) == 2


def test_dcp_cli_option_uses_engine_config_field_name():
    from lmdeploy.cli.utils import ArgumentHelper

    parser = argparse.ArgumentParser()
    ArgumentHelper.dcp(parser)
    assert parser.parse_args([]).dcp == 1
    assert parser.parse_args(['--dcp', '4']).dcp == 4


@pytest.mark.parametrize(('tp', 'dcp', 'rank', 'expected'),
                         [(4, 2, 0, (0, 1)), (4, 2, 3, (2, 3)),
                          (8, 2, 5, (4, 5)), (8, 4, 6, (4, 5, 6, 7)),
                          (8, 8, 7, tuple(range(8)))])
def test_dcp_group_membership(monkeypatch, tp, dcp, rank, expected):
    from lmdeploy.pytorch import distributed

    created = []

    def new_group(*, ranks, timeout, backend):
        group = (backend, tuple(ranks))
        created.append(group)
        return group

    monkeypatch.setattr(distributed.dist, 'new_group', new_group)
    context = SimpleNamespace(rank=rank,
                              dist_config=DistConfig(tp=tp, dcp=dcp),
                              attn_tp_group=SimpleNamespace(rank=rank % tp))
    distributed._build_dcp_group(context, timedelta(seconds=1))
    monkeypatch.setattr(
        distributed, 'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: context))

    assert context.dcp_group.rank == rank % dcp
    assert context.dcp_group.gpu_group == ('nccl', expected)
    assert context.dcp_group.cpu_group == ('gloo', expected)
    assert len(created) == 2 * (tp // dcp)
    assert distributed.get_dcp_world_rank() == (dcp, rank % dcp)


def test_dcp_group_teardown(monkeypatch):
    from lmdeploy.pytorch import distributed

    destroyed = []
    monkeypatch.setattr(distributed.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(distributed.dist, 'destroy_process_group',
                        destroyed.append)
    group = distributed.DistGroup(cpu_groups=['cpu0', 'cpu1'],
                                  gpu_groups=['gpu0', 'gpu1'])
    group.close()

    assert destroyed == ['cpu0', 'cpu1', 'gpu0', 'gpu1']
    assert group.cpu_groups is None and group.gpu_groups is None


class _DCPModelConfig(SimpleNamespace):

    @property
    def use_mla_fp8_cache(self):
        return self.mla_kv_cache_dtype == 'fp8_ds_mla'


def _valid_dcp_validation_inputs():
    model_config = _DCPModelConfig(
        hf_config=SimpleNamespace(
            model_type='glm_moe_dsa',
            architectures=['GlmMoeDsaForCausalLM'],
        ),
        use_flash_mla=True,
        mla_kv_cache_dtype='bfloat16',
        mla_index_topk=2048,
        num_replicate_key_value_heads=4,
        sliding_window=-1,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(max_batches=4,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=100,
                               dcp=2,
                               role=EngineRole.Hybrid)
    return model_config, cache_config, DistConfig(tp=4, dcp=2), MiscConfig()


@pytest.mark.parametrize('quant_policy', [QuantPolicy.NONE, QuantPolicy.FP8])
def test_validate_supported_glm_dcp_configuration(quant_policy):
    from lmdeploy.pytorch.engine.executor import _validate_dcp_config

    args = _valid_dcp_validation_inputs()
    args[1].quant_policy = quant_policy
    _validate_dcp_config(*args, specdecode_config=None, device_type='cuda')
    assert not args[0].use_mla_fp8_cache


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [('model_type', 'other', 'GlmMoeDsaForCausalLM'),
     ('mla_index_topk', 1024, 'top-k 512 or 2048'),
     ('quant_policy', QuantPolicy.INT4, 'quant_policy'),
     ('role', EngineRole.Decode, 'disaggregation')])
def test_validate_rejects_unsupported_dcp_modes(field, value, message):
    from lmdeploy.pytorch.engine.executor import _validate_dcp_config

    model, cache, dist, misc = _valid_dcp_validation_inputs()
    if field == 'model_type':
        model.hf_config.model_type = value
    elif field == 'quant_policy':
        cache.quant_policy = value
    elif field == 'role':
        cache = replace(cache, role=value)
    else:
        setattr(model, field, value)

    with pytest.raises(ValueError, match=message):
        _validate_dcp_config(model, cache, dist, misc, None, 'cuda')


@pytest.mark.parametrize(('rank', 'expected_final', 'expected_rows'),
                         [(0, [3, 4], [2, 3, 3, 4, 4]),
                          (1, [2, 4], [2, 2, 3, 3, 4])])
def test_nsa_metadata_localizes_each_causal_row(rank, expected_final,
                                                expected_rows):
    from lmdeploy.pytorch.backends.nsa import build_nsa_index_meta

    q_seqlens = torch.tensor([2, 3], dtype=torch.int32)
    kv_seqlens = torch.tensor([5, 8], dtype=torch.int32)
    cu_q = torch.nn.functional.pad(q_seqlens.cumsum(0), (1, 0))
    sequence_metadata = SimpleNamespace(
        q_seqlens=q_seqlens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=cu_q,
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        max_kv_seqlen=8,
        kv_flatten_size=13,
    )
    meta = build_nsa_index_meta(num_tokens=5,
                                is_decoding=False,
                                block_size=64,
                                num_gpu_blocks=4,
                                sequence_metadata=sequence_metadata,
                                dcp_world_rank=(2, rank))

    assert meta.dcp_k_seqlens.tolist() == expected_final
    assert meta.indexer_kv_seqlens.tolist() == expected_rows


def test_dcp_prefill_scoring_uses_global_sparse_boundary():
    from lmdeploy.pytorch.backends.cuda.nsa import TritonNSAIndexFP8
    from lmdeploy.pytorch.backends.nsa import build_nsa_index_meta

    q_seqlens = torch.tensor([2], dtype=torch.int32)
    kv_seqlens = torch.tensor([3000], dtype=torch.int32)
    sequence_metadata = SimpleNamespace(
        q_seqlens=q_seqlens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        block_offsets=torch.zeros(1, 1, dtype=torch.int32),
        max_kv_seqlen=3000,
        kv_flatten_size=3000,
    )
    meta = build_nsa_index_meta(num_tokens=2,
                                is_decoding=False,
                                block_size=64,
                                num_gpu_blocks=100,
                                sequence_metadata=sequence_metadata,
                                dcp_world_rank=(2, 0))
    impl = object.__new__(TritonNSAIndexFP8)
    impl.topk = 2048
    impl._allow_short_prefill_scoring_skip = True

    assert meta.max_kv_seqlen == 1500
    assert meta.global_max_kv_seqlen == 3000
    assert not impl._should_skip_scoring(meta)


def test_cudagraph_replay_refreshes_stable_dcp_lengths(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
    from lmdeploy.pytorch.models.utils import cudagraph as cudagraph_module
    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin

    monkeypatch.setattr(cudagraph_module, 'get_dcp_world_rank', lambda: (2, 1))
    model = CudaGraphMixin()
    graph_meta = CudaGraphMeta(max_batchs=4,
                               max_tokens=4,
                               num_blocks=2,
                               is_decoding=True,
                               device=torch.device('cpu'))
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.zeros(2, 2, dtype=torch.int32),
        q_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        q_seqlens=torch.ones(2, dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 8], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5, 13], dtype=torch.int32),
    )
    input_ids = torch.tensor([[1, 2]])
    position_ids = torch.tensor([[4, 7]])
    graph_meta.input_buffers = model.make_buffers_cudagraph(
        graph_meta, input_ids, position_ids, [], metadata)

    model.fill_buffers_cudagraph(graph_meta,
                                 input_ids,
                                 position_ids, [],
                                 metadata,
                                 inputs_embeds=None)
    local_buffer = graph_meta.input_buffers['dcp_kv_seqlens']
    buffer_ptr = local_buffer.data_ptr()
    assert local_buffer.tolist() == [2, 4, 0, 0]

    metadata.block_offsets = torch.zeros(2, 2, dtype=torch.int32)
    metadata.q_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    metadata.q_seqlens = torch.ones(2, dtype=torch.int32)
    metadata.kv_seqlens = torch.tensor([6, 9], dtype=torch.int32)
    model.fill_buffers_cudagraph(graph_meta,
                                 input_ids,
                                 position_ids, [],
                                 metadata,
                                 inputs_embeds=None)
    assert metadata.dcp_kv_seqlens.data_ptr() == buffer_ptr
    assert metadata.dcp_kv_seqlens.tolist() == [3, 4, 0, 0]
