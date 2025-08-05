# Copyright (c) OpenMMLab. All rights reserved.
# adpated from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py

from typing import Optional

import torch
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn.linear import build_qkv_proj, build_rowwise_linear
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1)**-0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask[:, :, :, :key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[WhisperConfig] = None,
        ctx_mgr: StepContextManager = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}'
                             f' and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f'Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and '
                'will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` '
                'when creating this class.')
        self.layer_idx = layer_idx

        # packed qkv
        # TODO, zhouxinyu, Whisper hard-code k_proj bias = False, may check correctness later
        self.qkv_proj = build_qkv_proj(self.embed_dim,
                                       num_q_heads=self.num_heads,
                                       num_kv_heads=self.num_heads,
                                       head_size=self.head_dim,
                                       bias=bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # o_proj
        self.out_proj = build_rowwise_linear(self.embed_dim,
                                             self.embed_dim,
                                             bias=bias,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_head_mask: torch.Tensor,
                output_attentions: bool = False,
                **kwargs):
        """Input shape: Batch x Time x Channel."""

        # qkv proj
        batch_size, q_len, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        query_states = query_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = query_states * self.scaling

        # attention
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    # def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #     """forward."""
    #     # qkv proj
    #     qkv_states = self.qkv_proj(hidden_states)
    #     q, k, v = self.qkv_proj.split_qkv(qkv_states)

    #     q = q.transpose(1, 2)
    #     k = k.transpose(1, 2)
    #     v = v.transpose(1, 2)
    #     q = q * self.scaling
    #     import pdb; pdb.set_trace()

    #     attn_output = nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)

    #     # o proj
    #     attn_output = attn_output.transpose(1, 2)
    #     attn_output = attn_output.flatten(-2, -1)
    #     attn_output = self.out_proj(attn_output)
    #     return attn_output, None


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):

    def __init__(self,
                 config: WhisperConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            ctx_mgr=ctx_mgr,
            dtype=dtype,
            device=device,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, dtype=dtype, device=device)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim, dtype=dtype, device=device)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights
