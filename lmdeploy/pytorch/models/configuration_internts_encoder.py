# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Union

from transformers import WhisperConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InternTimeSeriesEncoderConfig(WhisperConfig):

    model_type = 'intern_time_series_encoder'

    def __init__(self,
                 ts_adapt_in_dim: int = 256,
                 ts_adapt_out_dim: int = 1024,
                 ts_hidden_dim: int = 1024,
                 ts_cnn_channels: list[int] = [1, 32, 64, 128, 128],
                 ts_cnn_kernel_sizes: list[int] = [3, 5, 5, 5],
                 ts_cnn_strides: list[int] = [2, 4, 4, 5],
                 ts_cnn_paddings: list[int] = [1, 2, 2, 2],
                 ts_concat_subsampling_in_channels: int = 128,
                 ts_concat_subsampling_concat_size: int = 2,
                 use_flash_attn: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.ts_cnn_channels = ts_cnn_channels
        self.ts_cnn_kernel_sizes = ts_cnn_kernel_sizes
        self.ts_cnn_strides = ts_cnn_strides
        self.ts_cnn_paddings = ts_cnn_paddings
        self.ts_concat_subsampling_in_channels = ts_concat_subsampling_in_channels
        self.ts_concat_subsampling_concat_size = ts_concat_subsampling_concat_size

        self.ts_adapt_in_dim = ts_adapt_in_dim
        self.ts_adapt_out_dim = ts_adapt_out_dim

        self.ts_hidden_dim = ts_hidden_dim
        self.use_flash_attn = use_flash_attn

        assert self.ts_adapt_out_dim == self.ts_hidden_dim, 'ts_adapt_out_dim should be equal to ts_hidden_dim'
        assert self.ts_concat_subsampling_in_channels == self.ts_cnn_channels[
            -1], 'ts_concat_subsampling_in_channels should be equal to the out_channel of the last cnn layer'

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'ts_encoder_config' in config_dict:
            config_dict = config_dict['ts_encoder_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.')

        return cls.from_dict(config_dict, **kwargs)
