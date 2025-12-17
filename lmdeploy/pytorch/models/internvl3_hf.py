# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.models.utils.micro_batch import enable_micro_batch, split_batch
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import LayerNorm, RMSNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_o_proj, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import build_model_from_hf_config
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model
from .whisper import WhisperEncoderLayer


@torch.compile(dynamic=True)
def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


@torch.compile(dynamic=True)
def post_rms_norm(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                  variance: torch.Tensor, eps: float, embed_dim: int, dtype: torch.dtype):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


class InternVLVisionPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_channels,
    height, width)` into the initial `hidden_states` (patch embeddings) of
    shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels,
                                    hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    dtype=dtype,
                                    device=device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                'Make sure that the channel dimension of the pixel values match with the one set in the configuration.')

        embeddings = self.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings


class InternVLVisionEmbeddings(nn.Module):
    """Intern vision embedding."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device))
        else:
            self.mask_token = None
        self.patch_embeddings = InternVLVisionPatchEmbeddings(config, dtype=dtype, device=device)

        self.num_positions = self.patch_embeddings.num_patches + 1

        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.empty(1, self.num_positions, self.embed_dim, dtype=dtype, device=device))
        else:
            self.position_embeddings = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        target_dtype = embeddings.dtype
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.float().reshape(1, sqrt_num_positions, sqrt_num_positions,
                                                          -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed,
                                        size=(new_height, new_width),
                                        mode='bicubic',
                                        align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).to(target_dtype)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embeddings(pixel_values)  # shape = [*, channel, width, height]
        batch_size = patch_embeds.shape[0]
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_token, patch_embeds], dim=1)
        if self.position_embeddings is not None:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
            embeddings = embeddings + position_embeddings
        return embeddings


NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': LayerNorm,
}


class InternVLVisionAttention(nn.Module):
    """Intern vl attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=config.attention_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.use_qk_norm = config.use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )
            self.k_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )

        self.scale = self.head_dim**-0.5

        # o_proj
        self.projection_layer = build_o_proj(self.embed_dim,
                                             self.embed_dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True,
                                             tp_align_size=self.head_dim)

    def pre_rms_norm(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Pre rms norm."""
        return pre_rms_norm(q, k)

    def post_rms_norm(self, q: torch.Tensor, k: torch.Tensor, variance: torch.Tensor, dtype: torch.dtype):
        """Post rms norm."""
        eps = self.config.layer_norm_eps
        return post_rms_norm(q, k, self.q_norm.weight, self.k_norm.weight, variance, eps, self.embed_dim, dtype)

    def qkv_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import lmdeploy.pytorch.distributed as dist
        q_shape = q.shape
        k_shape = k.shape
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)

        tp, _ = get_tp_world_rank()
        if tp == 1:
            q = self.q_norm(q).view(q_shape)
            k = self.k_norm(k).view(k_shape)
            return q, k

        # variance
        variance = self.pre_rms_norm(q, k)
        dist.all_reduce(variance)
        q, k = self.post_rms_norm(q, k, variance, q.dtype)
        q = q.view(q_shape)
        k = k.view(k_shape)

        return q, k

    def forward(self, hidden_states):
        """forward."""

        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        if self.use_qk_norm:
            q, k = self.qkv_norm(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.projection_layer(attn_output)
        return attn_output


class InternVLVisionMLP(nn.Module):
    """Intern vl mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()

        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.act = ACT2FN[config.hidden_act]

        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            dp_disable_tp=True,
        )

        self.fc2 = build_rowwise_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            dp_disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVLVisionLayer(nn.Module):
    """Intern vision layer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = getattr(config, 'norm_type', 'rms_norm')

        self.attention = InternVLVisionAttention(config, dtype=dtype, device=device)
        self.mlp = InternVLVisionMLP(config, dtype=dtype, device=device)
        self.layernorm_before = NORM2FN[self.norm_type](self.embed_dim,
                                                        eps=config.layer_norm_eps,
                                                        dtype=dtype,
                                                        device=device)
        self.layernorm_after = NORM2FN[self.norm_type](self.embed_dim,
                                                       eps=config.layer_norm_eps,
                                                       dtype=dtype,
                                                       device=device)

        self.lambda_1 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))
        self.lambda_2 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _attn(self, hidden_states):
        hidden_states = hidden_states + self.attention(self.layernorm_before(hidden_states).to(
            hidden_states[0].dtype)) * self.lambda_1
        return hidden_states

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _mlp(self, hidden_states):
        hidden_states = hidden_states + self.mlp(self.layernorm_after(hidden_states).to(
            hidden_states.dtype)) * self.lambda_2
        return hidden_states

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self._attn(hidden_states)
        hidden_states = self._mlp(hidden_states)
        return hidden_states


class InternVLVisionEncoder(nn.Module):
    """Intern vision encoder."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [InternVLVisionLayer(config, dtype=dtype, device=device) for idx in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
    ):
        """forward."""
        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layer):
            layer_outputs = encoder_layer(hidden_states, )
            hidden_states = layer_outputs
        return hidden_states


@vlm_model
class InternVLVisionModel(nn.Module):
    """Intern vision model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config

        self.embeddings = InternVLVisionEmbeddings(config, dtype=dtype, device=device)
        self.encoder = InternVLVisionEncoder(config, dtype=dtype, device=device)
        self.layernorm = None
        if not config.use_mean_pooling:
            self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        """forward."""
        assert pixel_values.dim() == 4
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = hidden_states
        if self.layernorm is not None:
            last_hidden_state = self.layernorm(hidden_states)

        return hidden_states, last_hidden_state


class InternVLMultiModalProjector(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        input_dim = config.vision_config.hidden_size * int(1 / config.downsample_ratio)**2
        self.layer_norm = LayerNorm(input_dim, eps=1e-5, dtype=dtype, device=device)

        quantization_config = getattr(config.text_config, 'quantization_config', None)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_1 = build_colwise_linear(
            input_dim,
            config.text_config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            dp_disable_tp=True,
        )

        self.linear_2 = build_rowwise_linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            dp_disable_tp=True,
        )

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternS1TimeSeriesEncoder(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
        self.conv2 = nn.Conv1d(self.embed_dim,
                               self.embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dtype=dtype,
                               device=device)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model, eps=1e-5, dtype=dtype, device=device)

        self.adapt_in = build_colwise_linear(
            in_features=config.ts_adapt_in_dim,
            out_features=80,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adapt_out = build_rowwise_linear(
            in_features=self.embed_dim,
            out_features=config.ts_adapt_out_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def _make_causal_mask(self,
                          input_ids_shape: torch.Size,
                          dtype: torch.dtype,
                          device: torch.device,
                          past_key_values_length: int = 0):
        """Make causal mask used for bi-directional self-attention."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _prepare_decoder_attention_mask(self, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        return combined_attention_mask

    def forward(self, input_features):
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len - embed_pos.shape[0]]

            embed_pos = nn.functional.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]

        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        attention_mask = self._prepare_decoder_attention_mask(input_shape, inputs_embeds, past_key_values_length)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(hidden_states, attention_mask)
            hidden_states = layer_outputs

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)

        return hidden_states


class InternS1TimeSeriesConcatSubsampling(nn.Module):

    def __init__(self, in_channels: int, concat_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size

    def forward(self, ts_signals: torch.Tensor, ts_lens: torch.Tensor):
        if ts_signals.shape[1] % 2 != 0:
            ts_signals = ts_signals[:, :-1, :]
        even_frames = ts_signals[:, ::2, :]
        odd_frames = ts_signals[:, 1::2, :]
        ts_signals = torch.cat((even_frames, odd_frames), dim=2)
        ts_lens = ts_lens // 2
        return ts_signals, ts_lens


class InternS1TimeSeriesFixPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=20000, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # TODO: zhouxinyu, hf forces float32 during init, but becomes bf16 during forward
        pe = pe.unsqueeze(0).transpose(0, 1).to(dtype=dtype, device=device)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x.clone()


class InternS1TimeSeriesMultiChannelAdaptiveSubsampling(nn.Module):

    def __init__(self,
                 hidden_dim=128,
                 nhead=8,
                 num_encoder_layers=1,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=hidden_dim,
                              kernel_size=5,
                              stride=1,
                              padding=2,
                              dtype=dtype,
                              device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dtype=dtype, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pos_encoder = InternS1TimeSeriesFixPositionalEncoding(d_model=hidden_dim, dtype=dtype, device=device)
        self.subsampling = InternS1TimeSeriesConcatSubsampling(128, 2)

    def forward(self, inputs, input_lens, sr):
        sr = torch.as_tensor(sr, dtype=torch.float32)
        strides = torch.floor(160 / ((1 + torch.exp(-sr / 100))**6))
        patch_sizes = strides * 2
        patched_outputs = []
        output_lens = []

        for i in range(len(inputs)):
            seq = inputs[i]  # [seq_len, num_channel]
            ps = patch_sizes[i].item()
            st = strides[i].item()
            le = input_lens[i]

            output_len = torch.ceil((le - ps) / st) + 1
            pad_len = ((output_len - 1) * st + ps - le).long().item()
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq = nn.functional.pad(seq, (0, 0, 0, pad_len), 'constant', 0)
            assert output_len > 0, (seq.shape, ps, st, le, output_len)
            output_lens.append(output_len)
            indices = (torch.arange(0, output_len * st, st).unsqueeze(1) + torch.arange(ps)).long()
            patched = seq[indices]

            output = self.forward_encoder(patched)  # [num_patch, D]
            patched_outputs.append(output)

        outputs = nn.utils.rnn.pad_sequence(patched_outputs, batch_first=True)
        output_lens = torch.tensor(output_lens).squeeze().to(outputs.device).long()
        if output_lens.ndim == 0:
            output_lens = output_lens.unsqueeze(0)

        outputs, output_lens = self.subsampling(outputs.clone(), output_lens.clone())
        return outputs, output_lens

    def forward_encoder(self, x):
        num_patch, patch_len, C = x.shape
        # conv1
        # treat each channel as an independent sample and feed it into conv1
        x = x.reshape(num_patch * C, 1, patch_len)
        x = nn.functional.relu((self.conv(x)))  # [B*C, D1, L]
        x = x.permute(2, 0, 1)  # [L, B*C, D1]

        x = self.pos_encoder(x)  # [L, B*C, D1]
        x = self.transformer_encoder(x)
        x = x.mean(0)

        x = x.reshape(num_patch, C, -1)

        return x.mean(1)


class InternS1TimeSeriesModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.encoder_embed = InternS1TimeSeriesMultiChannelAdaptiveSubsampling(dtype=dtype, device=device)
        self.encoder = InternS1TimeSeriesEncoder(config, dtype=dtype, device=device)

    def forward(
        self,
        time_series_signals: Optional[torch.FloatTensor] = None,
        ts_lens: Optional[torch.Tensor] = None,
        sr: Optional[torch.Tensor] = None,
        time_series_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple]:
        if time_series_signals is None and time_series_embeds is None:
            raise ValueError('You have to specify time_series_signals or time_series_embeds')

        # embedded values can be passed in directly, but the dimensions must match
        if time_series_embeds is not None and len(
                time_series_embeds.shape) == 3 and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim:
            time_series_embeds = time_series_embeds
        else:
            if ((isinstance(time_series_signals, list) and len(time_series_signals[0].shape) == 2)
                    or (isinstance(time_series_signals, torch.Tensor) and len(time_series_signals.shape) == 3)):
                time_series_embeds, ts_lens = self.encoder_embed(time_series_signals, ts_lens, sr)
            else:
                raise ValueError(f'wrong time_series_signals size: {time_series_signals[0].shape}')

        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        last_hidden_state = self.encoder(input_features=time_series_embeds)

        # ts_lens after encoder
        ts_lens = (ts_lens + 1) // 2
        assert torch.all(ts_lens > 0), f'The length of time_series_embeds is so small. ts_lens: {ts_lens}'

        return last_hidden_state


class InternS1TimeSeriesProjector(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_norm = LayerNorm(config.ts_config.ts_hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.linear_1 = build_colwise_linear(in_features=config.ts_config.ts_hidden_dim,
                                             out_features=config.text_config.hidden_size,
                                             bias=True,
                                             dtype=dtype,
                                             device=device)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = build_rowwise_linear(in_features=config.text_config.hidden_size,
                                             out_features=config.text_config.hidden_size,
                                             bias=True,
                                             dtype=dtype,
                                             device=device)

    def forward(self, ts_features):
        hidden_states = self.layer_norm(ts_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternVLForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.vision_tower = InternVLVisionModel(config.vision_config, dtype=dtype, device=device)
        self.ts_tower = InternS1TimeSeriesModel(config.ts_config, dtype=dtype, device=device)
        self.multi_modal_projector = InternVLMultiModalProjector(config, dtype=dtype, device=device)
        self.time_series_projector = InternS1TimeSeriesProjector(config, dtype=dtype, device=device)
        self.language_model = build_model_from_hf_config(config.text_config, dtype=dtype, device=device)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

        self.input_processor = InternVLProcessor(self.config, dtype)

        self.compile_vit = False

    def compile_model(self):
        torch_version = version.parse(torch.__version__)
        if torch_version < version.parse('2.5.0'):
            return

        tp, _ = get_tp_world_rank()
        if torch_version >= version.parse('2.6.0') and tp > 1:
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            if isinstance(self.vision_tower, InternVLVisionModel):
                self.vision_tower.encoder.forward = split_batch(self.vision_tower.encoder.forward,
                                                                'inputs_embeds',
                                                                index=0)

        self.get_image_features = torch.compile(self.get_image_features, mode='max-autotune-no-cudagraphs')
        self.compile_vit = True
        self.has_compiled_vit = False

    def _mark_dynamic_once(self, pixel_values, dims):
        """Call torch._dynamo.mark_dynamic to avoid recompile."""
        if not self.compile_vit or self.has_compiled_vit or pixel_values is None:
            return

        torch._dynamo.mark_dynamic(pixel_values, dims)
        self.has_compiled_vit = True

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """Obtains image last hidden states from the vision tower and apply
        multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int` or `List[int]`):
                Layer index or list of layer indices to extract features from.
        Returns:
            vision_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        downsample_ratio = self.config.downsample_ratio
        hidden_states, last_hidden_state = self.vision_tower(pixel_values=pixel_values)
        if vision_feature_layer == -1:
            vision_features = last_hidden_state
        else:
            vision_features = hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == 'default':
            vision_features = vision_features[:, 1:, :]

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)

        return vision_features

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor,
                                                width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError('Height and width must be divisible by scale_factor for proper downsampling.')

        # Reshape to allow downsampling
        vision_features = vision_features.view(batch_size, width, int(height * scale_factor),
                                               int(channels / scale_factor))
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(batch_size, int(height * scale_factor), int(width * scale_factor),
                                               int(channels / (scale_factor**2)))

        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    def get_ts_feature(self, ts_values, ts_lens, sr):
        ts_embeds = self.ts_tower(time_series_signals=ts_values, ts_lens=ts_lens, sr=sr)
        ts_embeds = self.time_series_projector(ts_embeds)
        return ts_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        ts_values: torch.Tensor = None,
        ts_lens: torch.Tensor = None,
        ts_sr: torch.Tensor = None,
        ts_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if inputs_embeds is None and pixel_values is not None:
            # extract feature
            self._mark_dynamic_once(pixel_values, [0])
            vit_embeds = self.get_image_features(
                pixel_values,
                self.vision_feature_layer,
                self.vision_feature_select_strategy,
            )
            lang_embeds = self.get_input_embeddings()(input_ids)
            lang_embeds.masked_scatter_(image_mask[..., None], vit_embeds)

            inputs_embeds = lang_embeds
            input_ids = None
        elif inputs_embeds is None and ts_values is not None:
            ts_features = self.get_ts_feature(ts_values, ts_lens, ts_sr)  # [B, T, C]
            lang_embeds = self.get_input_embeddings()(input_ids)
            lang_embeds.masked_scatter_(ts_mask[..., None], ts_features)

            inputs_embeds = lang_embeds
            input_ids = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        outputs = self.language_model.forward(input_ids=input_ids,
                                              inputs_embeds=inputs_embeds,
                                              past_key_values=past_key_values,
                                              position_ids=position_ids,
                                              attn_metadata=attn_metadata)
        return outputs

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = None

        # vision inputs
        pixel_values = None
        image_mask = None
        ts_values = None
        ts_lens = None
        ts_sr = None
        ts_mask = None
        if context.input_multimodals is not None:
            mm_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            mm_values = [data for im_data in mm_values for data in im_data]

            if len(mm_values) > 0:
                is_time_series = mm_values[0].meta.get('ts_token_id', False)
                if is_time_series:
                    ts_values = mm_values
                    ts_token_id = ts_values[0].meta['ts_token_id']
                    ts_lens = ts_values[0].meta['ts_lens']
                    ts_sr = ts_values[0].meta['ts_sr']
                    ts_mask = input_ids == ts_token_id
                    ts_values = torch.cat([data.data for data in ts_values])
                else:
                    pixel_values = mm_values
                    image_token_id = pixel_values[0].meta['image_token_id']
                    image_mask = input_ids == image_token_id
                    pixel_values = torch.cat([data.data for data in pixel_values])

        # get inputs from context
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            vision_embedding_indexing = context.input_embedding_indexing
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_mask=image_mask,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            ts_mask=ts_mask,
            inputs_embeds=inputs_embeds,
        )

    def load_lora_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], adapter_id: int):
        """Load lora weights."""

        if hasattr(self.model.language_model, 'load_lora_weights'):
            return self.model.language_model.load_lora_weights(weights, adapter_id)
        else:
            from lmdeploy.pytorch.adapter.adapter import load_lora_weights

            return load_lora_weights(weights, adapter_id)

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        if name == 'lm_head.weight':
            return 'language_model.lm_head.weight'
        elif name.startswith('model.language_model.'):
            return 'language_model.model.' + name[len('model.language_model.'):]
        elif name.startswith('model.'):
            return name[len('model.'):]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        lang_prefix = 'language_model.'
        lang_prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        vision_stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        for name, loaded_weight in weights:

            if name.startswith(lang_prefix):
                new_key = name[lang_prefix_length:]
                new_weights[new_key] = loaded_weight
                continue

            for (param_name, weight_name, shard_id) in vision_stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
                    continue
                elif name in buffers_dict:
                    param = buffers_dict[name]
                    load_weight(param, loaded_weight)
                    continue

        self.language_model.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class InternVLProcessor(BaseModelInputProcessor):
    """Internvl input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            if 'ts_values' in input_mm:
                ts_values = input_mm['ts_values'].to(self.dtype)
                offset = input_mm['offset']
                ts_token_id = input_mm['ts_token_id']
                ts_lens = input_mm['ts_lens']
                ts_sr = input_mm['ts_sr']
                num_pad = input_mm['num_ts_tokens']

                mm_data = MultiModalTensor(data=ts_values,
                                           start=offset,
                                           end=offset + num_pad,
                                           meta=dict(ts_token_id=ts_token_id, ts_lens=ts_lens, ts_sr=ts_sr))
            else:
                pixel_values = input_mm['pixel_values'].to(self.dtype)
                offset = input_mm['offset']
                image_token_id = input_mm['image_token_id']
                num_pad = input_mm['image_tokens']
                if isinstance(num_pad, torch.Tensor):
                    num_pad = num_pad.item()

                mm_data = MultiModalTensor(data=pixel_values,
                                           start=offset,
                                           end=offset + num_pad,
                                           meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
