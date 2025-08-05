# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import WhisperPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.models.whisper import WhisperEncoderLayer
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .configuration_internts_encoder import InternTimeSeriesEncoderConfig


def _make_causal_mask(input_ids_shape: torch.Size,
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


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len,
    src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class InternTimeSeriesEncoder(WhisperPreTrainedModel):

    def __init__(self,
                 config: InternTimeSeriesEncoderConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config)

        self.config = config

        self.dropout = config.dropout
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

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
            for _ in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model, dtype=dtype, device=device)

        self.gradient_checkpointing = False
        self.post_init()

        self.mask_type = None
        self.chunk_length = None

        self.adapt_in = nn.Linear(config.ts_adapt_in_dim, 80, dtype=dtype, device=device)
        self.adapt_out = nn.Linear(self.embed_dim, config.ts_adapt_out_dim, dtype=dtype, device=device)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def define_masktype(self, masktype, chunk_length=None):
        self.mask_type = masktype
        self.chunk_length = chunk_length

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       combined_attention_mask)
        return combined_attention_mask

    def prepare_chunk_attention_mask(self, attention_mask, input_shape, inputs_embeds):

        block_size = round(self.chunk_length / 4 * 2)
        matrix_size = input_shape[1]

        matrix = torch.ones(matrix_size, matrix_size)

        num_full_blocks = round(matrix_size // block_size)
        remainder = matrix_size % block_size
        for i in range(num_full_blocks):
            row_start = i * block_size
            col_start = i * block_size
            matrix[row_start:row_start + block_size,
                   col_start:col_start + block_size] = torch.zeros(block_size, block_size)

        if remainder > 0:
            last_row_start = num_full_blocks * block_size
            last_col_start = num_full_blocks * block_size
            matrix[last_row_start:last_row_start + remainder,
                   last_col_start:last_col_start + remainder] = torch.zeros(remainder, remainder)

        matrix = matrix * -65504
        matrix = matrix.unsqueeze(0).unsqueeze(0).repeat(input_shape[0], 1, 1, 1)
        attention_mask = matrix.to(inputs_embeds.device)
        return attention_mask

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # torch.Size([1, 100, 768])
        embed_pos = self.embed_positions.weight  # torch.Size([1500, 768])

        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len - embed_pos.shape[0]]

            embed_pos = F.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        attention_mask = None
        if self.mask_type == 'chunk':
            attention_mask = self.prepare_chunk_attention_mask(attention_mask, input_shape, inputs_embeds)
        else:
            attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds,
                                                                  past_key_values_length)

        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (self.layer_norm(hidden_states), )
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):

                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class ConcatSubsampling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        concat_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):

        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]

        even_frames = x[:, ::2, :]
        odd_frames = x[:, 1::2, :]
        x = torch.cat((even_frames, odd_frames), dim=2)

        x_lens = x_lens // 2
        return x, x_lens


class Conv1dSubsampling(nn.Module):

    def __init__(self, config: InternTimeSeriesEncoderConfig):
        super().__init__()

        self.channels = config.ts_cnn_channels
        self.kernel_sizes = config.ts_cnn_kernel_sizes
        self.strides = config.ts_cnn_strides
        self.paddings = config.ts_cnn_paddings
        self.concat_in_channels = config.ts_concat_subsampling_in_channels
        self.concat_size = config.ts_concat_subsampling_concat_size

        self.n_layers = len(self.channels) - 1
        if not (len(self.kernel_sizes) == len(self.strides) == len(self.paddings) == self.n_layers):
            raise ValueError('Length mismatch: channels should have 1 more element than kernel_sizes/strides/paddings.')

        self.layers = nn.ModuleList()
        in_channels = self.channels[0]
        for i in range(self.n_layers):
            out_channels = self.channels[i + 1]
            conv = nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=self.kernel_sizes[i],
                             stride=self.strides[i],
                             padding=self.paddings[i])
            self.layers.append(nn.Sequential(conv))
            in_channels = out_channels

        self.subsampling = ConcatSubsampling(self.concat_in_channels, self.concat_size)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):

        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        x = x.transpose(1, 2)

        x_lens = ((((x_lens + 1) // 2 + 3) // 4 + 3) // 4 + 4) // 5
        x, x_lens = self.subsampling(x, x_lens)
        return x, x_lens


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = lengths.max()
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(-1)


@dataclass
class TimeSeriesModelOutput(BaseModelOutput):
    ts_pad_mask: Optional[torch.FloatTensor] = None


class InternTimeSeriesModel(PreTrainedModel):
    main_input_name = 'time_series_signals'
    _supports_flash_attn_2 = False
    config_class = InternTimeSeriesEncoderConfig
    _no_split_modules = ['WhisperEncoderLayer']

    def __init__(self,
                 config: InternTimeSeriesEncoderConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config)
        self.config = config

        self.encoder_embed = Conv1dSubsampling(config)
        self.encoder = InternTimeSeriesEncoder(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        # import pdb; pdb.set_trace()

    def get_input_embeddings(self):
        return self.encoder_embed

    def forward(
        self,
        time_series_signals: Optional[torch.FloatTensor] = None,
        x_lens: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        time_series_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, TimeSeriesModelOutput]:

        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if time_series_signals is None and time_series_embeds is None:
            raise ValueError('You have to specify time_series_signals or time_series_embeds')

        # we can directly pass the embedded value, but dimensions must match
        if time_series_embeds is not None and len(
                time_series_embeds.shape) == 3 and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim:
            input_embeds = time_series_embeds
        else:
            if len(time_series_signals.shape) == 3:
                # x_lens should be // 10, then // 2 in the encoder, totaling a division by 20 from the CNN.
                input_embeds, x_lens = self.encoder_embed(time_series_signals, x_lens)
            else:
                raise ValueError(f'wrong time_series_signals size: {time_series_signals.shape}')

        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        encoder_outputs = self.encoder(
            input_features=input_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # x_lens after encoder
        x_lens = (x_lens + 1) // 2
        assert torch.all(x_lens > 0), f'The length of time_series_embeds is so small. x_lens: {x_lens}'

        src_key_padding_mask = make_pad_mask(x_lens)

        last_hidden_state = encoder_outputs.last_hidden_state
        if not return_dict:
            return (last_hidden_state, ) + encoder_outputs[1:]

        return TimeSeriesModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,  # encoder_state，for every layer
            attentions=encoder_outputs.attentions,
            ts_pad_mask=src_key_padding_mask)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                print(f'=> load packed weights: {name}')
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
                print(f'=> load weights: {name}')
