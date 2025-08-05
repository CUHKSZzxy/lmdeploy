# Copyright (c) OpenMMLab. All rights reserved.
import base64
import io
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
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
class InternVL3VisionModel(VisonModel):
    """Internvl3 vision model."""

    # FIXME: hack for now
    _arch = ['InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration', 'InternTSChatModel']

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        self.arch = hf_config.architectures[0]
        self.arch_type = 'audio'  # FIXME, determine visual or audio from hf_config

    # def build_audio_processor(self):
    #     def load_time_series_signals_from_file(file_path: str):
    #         """Load time series signals from a file path."""
    #         try:
    #             import librosa  # noqa: F401
    #         except ImportError:
    #             raise ImportError('To use InternS1-Mini, please install audio dependencies by `pip install librosa`')

    #         # Load audio directly from file path
    #         ts_input, _ = librosa.load(file_path, sr=None)
    #         ts_input_len = torch.tensor([len(ts_input)])
    #         ts_output_len = (((((len(ts_input) + 1) // 2 + 3) // 4 + 3)// 4 + 3)// 10 + 1)// 2
    #         ts_input = torch.from_numpy(ts_input).float()
    #         return ts_input, ts_input_len, ts_output_len

    #     def load_time_series_signals_from_base64(base64_data: str):
    #         try:
    #             audio_bytes = base64.b64decode(base64_data)
    #             return load_time_series_signals_from_bytes(audio_bytes)
    #         except Exception as e:
    #             raise ValueError(f"Cannot decode base64 data: {str(e)}")

    #     def load_time_series_signals_from_bytes(audio_bytes: bytes):
    #         """Load time series signals directly from byte data, avoiding temporary file I/O"""
    #         try:
    #             import librosa  # noqa: F401
    #         except ImportError:
    #             raise ImportError('To use InternS1-Mini, please install audio dependencies by `pip install librosa`')

    #         # Convert byte data to a file-like object
    #         audio_file_like = io.BytesIO(audio_bytes)

    #         # Load audio directly from byte data in memory
    #         ts_input, _ = librosa.load(audio_file_like, sr=None)
    #         ts_input_len = torch.tensor([len(ts_input)])
    #         ts_output_len = (((((len(ts_input) + 1) // 2 + 3) // 4 + 3)// 4 + 3)// 10 + 1)// 2
    #         ts_input = torch.from_numpy(ts_input).float()
    #         return ts_input, ts_input_len, ts_output_len

    #     def auto_load_audio(input_data: str):
    #         """Auto-detect the input type and load accordingly."""
    #         if input_data.startswith('data:audio/'):
    #             # Base64 encoded audio data
    #             base64_data = input_data.split(',')[1]
    #             return load_time_series_signals_from_base64(base64_data)
    #         elif input_data.endswith('.wav') or input_data.endswith('.mp3'):
    #             # File path to audio file
    #             return load_time_series_signals_from_file(input_data)
    #         else:
    #             raise ValueError(f"Unsupported audio input format: {input_data}")

    #     return auto_load_audio
    #     # return load_time_series_signals_from_base64

    def build_audio_processor(self):
        """Builds a processor that can load audio from a file path or a base64
        data URI."""

        def _process_audio_array(ts_input_np):
            ts_input_len = torch.tensor([len(ts_input_np)])
            # FIXME, what's the magic number here ???
            ts_output_len = (((((len(ts_input_np) + 1) // 2 + 3) // 4 + 3) // 4 + 3) // 10 + 1) // 2
            ts_input = torch.from_numpy(ts_input_np).float()
            return ts_input, ts_input_len, ts_output_len

        def auto_load_audio(input_data: str):
            try:
                import librosa
            except ImportError:
                raise ImportError('To use audio model, please install audio dependencies by `pip install librosa`')

            audio_source = None
            if input_data.startswith('data:audio/'):
                try:
                    base64_data = input_data.split(',')[1]
                    audio_bytes = base64.b64decode(base64_data)
                    audio_source = io.BytesIO(audio_bytes)
                except Exception as e:
                    raise ValueError(f'Cannot decode base64 data: {str(e)}')
            elif input_data.endswith(('.wav', '.mp3')):
                audio_source = input_data
            else:
                raise ValueError(f'Unsupported audio input format: {input_data}')

            ts_input_np, _ = librosa.load(audio_source, sr=None)
            return _process_audio_array(ts_input_np)

        return auto_load_audio

    def build_preprocessor(self):
        if self.arch_type == 'audio':
            self.processor = self.build_audio_processor()
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            tokenizer = self.processor.tokenizer
            self.image_token_id = tokenizer.context_image_token_id
            self.image_tokens_per_patch = self.processor.image_seq_length
            self.tokenizer_init_kwargs = tokenizer.init_kwargs

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

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        if self.arch_type == 'audio':
            # TODO: adding logics here
            audios = self.collect_audios(messages)
            audios = [self.processor(audio) for audio in audios]
            outputs = []
            for audio_values, audio_len, audio_output_len in audios:
                data = dict(audio_values=audio_values, audio_len=audio_len, audio_output_len=audio_output_len)
                outputs.append(data)
            messages.append(dict(role='preprocess', content=outputs))
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
        # FIXME, double check logic for audio here
        if self.arch_type == 'audio':
            inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
            inputs = inputs[0]
            assert all(x.get('audio_values') is not None for x in inputs)
            outputs = []
            for idx in range(0, len(inputs), max_batch_size):
                audio_values = [x['audio_values'] for x in inputs[idx:idx + max_batch_size]]
                audio_len = [x['audio_len'] for x in inputs[idx:idx + max_batch_size]]
                audio_output_len = [x['audio_output_len'] for x in inputs[idx:idx + max_batch_size]]
                audio_values = torch.stack(audio_values, dim=0).to(self.model.device, dtype=torch.float16)
                audio_len = torch.stack(audio_len, dim=0).to(self.model.device, dtype=torch.int32)
                audio_output_len = torch.stack(audio_output_len, dim=0).to(self.model.device, dtype=torch.int32)
                logger.info(f'audio forward shape: {audio_values.shape}')
                feats = self.model.get_audio_features(audio_values, audio_len, audio_output_len)
                outputs.extend(feats)
            messages.append(dict(role='forward', content=outputs))
        else:
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

    @staticmethod
    def proc_vl_messages(
        messages,
        chat_template,
        sequence_start,
        tools: Optional[List[object]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = content[0]
            if IMAGE_TOKEN in prompt and f'<img>{IMAGE_TOKEN}' not in prompt:
                prompt = prompt.replace(f'{IMAGE_TOKEN}', f'<img>{IMAGE_TOKEN}</img>')
                prompt = prompt.replace('</img><img>', '')
                prompt = prompt.replace('<img><img>', '<img>')
                prompt = prompt.replace('</img></img>', '</img>')
            elif IMAGE_TOKEN not in prompt:
                prompt = f'<img>{IMAGE_TOKEN * n_images}</img>\n' + prompt
            else:
                pass
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages,
                                               sequence_start,
                                               tools=tools,
                                               enable_thinking=enable_thinking)
        return prompt, IMAGE_TOKEN

    @staticmethod
    def proc_audio_messages(
        messages,
        chat_template,
        sequence_start,
        tools: Optional[List[object]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        TS_TOKEN = '<TS_CONTEXT>'
        # TS_START_TOKEN = '<ts>'
        # TS_END_TOKEN = '</ts>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['preprocess', 'forward']:
                continue

            # FIXME: should fix the prompt msg processing logics
            # but since the user concate the prompt too casually, we may require users to prepare the inputs themselves
            # maybe we should access the forward attribute of the message, to concatenate a valid prompt

            # n_audios = len([1 for x in message['content'] if x['type'] == 'audio'])
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = content[0]
            # if TS_TOKEN in prompt and f'<time_series_signals>{TS_TOKEN}' not in prompt:
            #     prompt = prompt.replace(f'{TS_TOKEN}', f'<time_series_signals>{TS_TOKEN}</time_series_signals>')
            #     prompt = prompt.replace('</time_series_signals><time_series_signals>', '')
            #     prompt = prompt.replace('<time_series_signals><time_series_signals>', '<time_series_signals>')
            #     prompt = prompt.replace('</time_series_signals></time_series_signals>', '</time_series_signals>')
            # elif TS_TOKEN not in prompt:
            #     prompt = f'<time_series_signals>{TS_TOKEN * n_audios}</time_series_signals>\n' + prompt
            # else:
            #     pass
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages,
                                               sequence_start,
                                               tools=tools,
                                               enable_thinking=enable_thinking)
        return prompt, TS_TOKEN

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: Optional[List[object]] = None,
                   enable_thinking: Optional[bool] = None,
                   **kwargs):
        if self.arch_type == 'audio':
            prompt, IMAGE_TOKEN = self.proc_audio_messages(messages,
                                                           chat_template,
                                                           sequence_start,
                                                           tools=None,
                                                           enable_thinking=None)
        else:
            prompt, IMAGE_TOKEN = self.proc_vl_messages(messages,
                                                        chat_template,
                                                        sequence_start,
                                                        tools=tools,
                                                        enable_thinking=enable_thinking)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     tools: Optional[List[object]] = None,
                     enable_thinking: Optional[bool] = None,
                     **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 enable_thinking=enable_thinking)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
