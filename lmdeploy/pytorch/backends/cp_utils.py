# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_dcp_local_seq_lens(seq_lens: torch.Tensor,
                           dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Return interleaved per-rank lengths for global sequence lengths."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return seq_lens
    numer = seq_lens + dcp_world_size - 1 - dcp_rank
    return torch.clamp(torch.div(numer, dcp_world_size, rounding_mode='floor'),
                       min=0)


def get_dcp_local_cu_seqlens(
        seq_lens: torch.Tensor,
        dcp_world_rank: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local lengths and their INT32 cumulative offsets."""
    local_lens = get_dcp_local_seq_lens(seq_lens, dcp_world_rank)
    cu_lens = torch.nn.functional.pad(
        torch.cumsum(local_lens, dim=0, dtype=torch.int32), (1, 0))
    return local_lens, cu_lens


def fill_dcp_local_seq_lens(seq_lens: torch.Tensor,
                            local_lens: torch.Tensor,
                            dcp_world_rank: tuple[int, int]) -> None:
    """Refresh a graph-stable local-length buffer without allocations."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if local_lens.shape != seq_lens.shape:
        raise ValueError('local_lens and seq_lens must have identical shapes')
    torch.add(seq_lens, dcp_world_size - 1 - dcp_rank, out=local_lens)
    torch.div(local_lens,
              dcp_world_size,
              rounding_mode='floor',
              out=local_lens)
    local_lens.clamp_min_(0)


def get_dcp_local_indices(indices: torch.Tensor,
                          dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Map global token positions to this rank's local positions."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return indices
    valid = (indices >= 0) & (indices % dcp_world_size == dcp_rank)
    local_indices = torch.div(indices.clamp_min(0),
                              dcp_world_size,
                              rounding_mode='floor')
    return torch.where(valid, local_indices, -1)


def compact_dcp_local_indices(
        indices: torch.Tensor,
        dcp_world_rank: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter global winners to this rank and compact valid local ids."""
    local_indices = get_dcp_local_indices(indices, dcp_world_rank)
    valid = local_indices >= 0
    valid_counts = valid.sum(dim=-1, dtype=torch.int32)
    # Stable partition keeps the global selector's order while moving invalid
    # slots behind all local winners. Shapes remain fixed for CUDA graphs.
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    return local_indices.gather(-1, order), valid_counts
