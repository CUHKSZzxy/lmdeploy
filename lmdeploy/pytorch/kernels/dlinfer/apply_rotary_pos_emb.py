# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Optional[Tensor],
    k_embed: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    query_states_reshaped = query_states.unsqueeze(0)
    key_states_reshaped = key_states.unsqueeze(0)
    query_states_reshaped, key_states_reshaped = \
        ext_ops.apply_rotary_pos_emb(query_states_reshaped,
                                     key_states_reshaped,
                                     cos, sin,
                                     None, None)
    if q_embed is None:
        q_embed = query_states_reshaped.view(query_states.shape)
    elif q_embed is not query_states:
        q_embed.copy_(query_states_reshaped.view(query_states.shape))

    if k_embed is None:
        k_embed = key_states_reshaped.view(key_states.shape)
    elif k_embed is not key_states:
        k_embed.copy_(key_states_reshaped.view(key_states.shape))

    return q_embed, k_embed
