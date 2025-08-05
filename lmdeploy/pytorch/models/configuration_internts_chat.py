# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from lmdeploy.pytorch.models.configuration_internts_encoder import InternTimeSeriesEncoderConfig

logger = logging.get_logger(__name__)


class InternTSChatConfig(PretrainedConfig):
    model_type = 'interts_chat'
    is_composition = True

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        ts_model_path: Optional[str] = None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        template='internts',
        adapter_type='mlp',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm_model_path = llm_model_path
        self.ts_model_path = ts_model_path

        if ts_model_path is None:
            ts_config = {'architectures': ['InternTimeSeriesModel']}
            self.ts_config = InternTimeSeriesEncoderConfig(**ts_config)
            logger.info('ts_config is None. Initializing the InternTimeSeriesConfig with default values.')
        else:
            self.ts_config = InternTimeSeriesEncoderConfig.from_pretrained(self.ts_model_path)

        if llm_model_path is None:
            llm_config = {'architectures': ['Qwen2ForCausalLM']}
            from transformers import Qwen2Config
            self.llm_config = Qwen2Config(**llm_config)
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`QWen2Config`).')
        else:
            self.llm_config = AutoConfig.from_pretrained(self.llm_model_path)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
        self.adapter_type = adapter_type
        logger.info(f'time_series_select_layer: {self.select_layer}')
        logger.info(f'time_series_model_path: {self.ts_model_path}')
        logger.info(f'llm_model_path: {self.llm_model_path}')
        logger.info(f'llm_model_architecture: {self.llm_config.architectures[0]}')
        logger.info(f'ts_model_architecture: {self.ts_config.architectures[0]}')

    def to_dict(self):
        """Serializes this instance to a Python dictionary. Override the
        default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = {
            'ts_model_path': self.ts_model_path,
            'llm_model_path': self.llm_model_path,
            'model_type': self.__class__.model_type,
            'use_backbone_lora': self.use_backbone_lora,
            'use_llm_lora': self.use_llm_lora,
            'select_layer': self.select_layer,
            'adapter_type': self.adapter_type,
            'template': self.template
        }

        return output
