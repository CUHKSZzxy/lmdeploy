# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VisionModel
from lmdeploy.vl.model.internvl import VISION_MODELS, InternVLVisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        'text_kwargs': {
            'padding': False,
        },
        'images_kwargs': {
            'crop_to_patches': True,
        },
        'videos_kwargs': {},
    }


@VISION_MODELS.register_module()
class InternVL3VisionModel(InternVLVisionModel):
    """Internvl3 vision model."""

    _arch = ['InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration']

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        self.arch = self.hf_config.architectures[0]

    def build_preprocessor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.context_image_token_id
        self.image_tokens_per_patch = self.processor.image_seq_length
        self.tokenizer_init_kwargs = tokenizer.init_kwargs

        self.ts_token = tokenizer.context_ts_token
        self.ts_token_id = tokenizer.context_ts_token_id
        self.end_ts_token = tokenizer.end_ts_token
        self.start_ts_token = tokenizer.start_ts_token
        self.start_ts_token_id = tokenizer.start_ts_token_id
        self.end_ts_token_id = tokenizer.end_ts_token_id

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights():
            if self.arch == 'InternVLForConditionalGeneration':
                model = AutoModel.from_config(self.hf_config, trust_remote_code=True)
                if not self.with_llm:
                    del model.language_model
            elif self.arch == 'InternS1ForConditionalGeneration':
                model = AutoModelForCausalLM.from_config(self.hf_config, trust_remote_code=True)
                if not self.with_llm:
                    del model.model.language_model
            else:
                raise ValueError(f'unsupported model arch {self.arch}')

        model.half()
        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map='auto' if not self.with_llm else {'': 'cpu'},
                                         max_memory=self.max_memory,
                                         no_split_module_classes=['InternVLVisionLayer', 'InternS1VisionLayer'],
                                         dtype=torch.half)
        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()

    def preprocess(self, messages: List[Dict], time_series_inputs: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""

        if time_series_inputs is not None:
            time_series_inputs = self.processor.time_series_processor(
                ts_paths=time_series_inputs['time_series_paths'],
                sampling_rates=time_series_inputs['time_series_sampling_rates'])
            time_series_inputs.update({'ts_token_id': self.ts_token_id})
            outputs = [time_series_inputs]
        else:
            from transformers.image_utils import make_flat_list_of_images
            output_kwargs = self.processor._merge_kwargs(
                InternVLProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer_init_kwargs,
                **{
                    'return_tensors': 'pt',
                    'add_special_tokens': False
                },
            )
            images = self.collect_images(messages)
            images = [image.convert('RGB') for image, _ in images]
            num_image = len(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.processor.image_processor(images, **output_kwargs['images_kwargs'])
            image_num_patches = image_inputs.pop('num_patches').cpu().numpy().tolist()
            image_pixel_values = image_inputs.pop('pixel_values')
            outputs = []
            cum_num_patches = 0
            for idx in range(num_image):
                cur_num_patches = image_num_patches[idx]
                pixel_values = image_pixel_values[cum_num_patches:cum_num_patches + cur_num_patches, ...]
                cum_num_patches += cur_num_patches
                data = dict(pixel_values=pixel_values,
                            image_tokens=self.image_tokens_per_patch * cur_num_patches,
                            image_token_id=self.image_token_id)
                outputs.append(data)

        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        assert all(x.get('pixel_values') is not None for x in inputs)
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            split = [x.shape[0] for x in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.model.device, dtype=torch.float16)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            feats = self.model.get_image_features(
                pixel_values,
                vision_feature_layer=self.hf_config.vision_feature_layer,
                vision_feature_select_strategy=self.hf_config.vision_feature_select_strategy,
            )
            feats = torch.split(feats, split, dim=0)
            outputs.extend([x.reshape(-1, x.shape[-1]) for x in feats])
        messages.append(dict(role='forward', content=outputs))
        return messages

    def has_time_series_data(self, messages):
        for message in messages:
            role, content = message['role'], message['content']

            if role == 'preprocess':
                content = message['content']
                has_ts_data = any(isinstance(item, dict) and 'ts_values' in item for item in content)
                return has_ts_data

        return False

    def proc_messages(
        self,
        messages,
        chat_template,
        sequence_start,
        tools: Optional[List[object]] = None,
        chat_template_kwargs: Optional[Dict] = None,
    ):
        chat_template_kwargs = chat_template_kwargs or {}
        """Apply chat template to get the prompt."""
        has_time_series_data = self.has_time_series_data(messages)

        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]

        if has_time_series_data:
            prompt_messages = messages
            prompt = chat_template.messages2prompt(prompt_messages, sequence_start, tools=tools, **chat_template_kwargs)
            return prompt, self.ts_token

        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(f'{IMAGE_TOKEN}', f'<img>{self.image_token}</img>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                _content = []
                for item in content:
                    item_type = item['type']
                    if item_type == 'text':
                        _content.append(item['text'])
                    elif item_type in ['image', 'image_url']:
                        _content.append(f'<img>{self.image_token}</img>\n')
                    else:
                        raise ValueError(f'Unsupported message type: {item["type"]}')
                prompt_messages.append(dict(role='user', content=''.join(_content)))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, tools=tools, **chat_template_kwargs)
        return prompt, self.image_token

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: Optional[List[object]] = None,
                   chat_template_kwargs: Optional[Dict] = None,
                   **kwargs):
        # FIXME: zhouxinyu, should rename IMAGE_TOKEN, since now we have time series token
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 chat_template_kwargs=chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
