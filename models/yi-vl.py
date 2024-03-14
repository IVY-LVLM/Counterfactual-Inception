import os
import io
import base64
import numpy as np
import requests

import torch

from tools.read_yaml import get_hf_home

from .base_model import BaseModel
from models.llava_yi.conversation import conv_templates
from models.llava_yi.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    load_pretrained_model,
    process_images,
    tokenizer_image_token,
)
from models.llava_yi.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

default_path = os.path.join(get_hf_home(), "hub/Yi-VL-6B")

def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")

class YiVL(BaseModel):
    def __init__(self, temperature, max_new_tokens, model_name: str = "yi-vl", model_path: str = default_path, conv_mode: str = "mm_default", counterfactual: bool = False,):
        super().__init__(model_name=model_name, model_path=model_path, counterfactual=counterfactual)
        model_path = os.path.expanduser(model_path)
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path)
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, counterfactual_keyword = ""):
        intermediate_output = ""
        raw_image_data = get_pil_image(raw_image_data)
        raw_image_data = raw_image_data.convert("RGB")
        image_tensor = process_images([raw_image_data], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)
        
        if self.counterfactual:
            formatted_prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(formatted_prompt) == 1:
                prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt[0]
                
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                stop_str = conv.sep
                
                outputs = self.generate_sentence(image_tensor, stop_str, prompt)
                intermediate_output = ""
            elif len(formatted_prompt) == 2:
                prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt[0]
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                stop_str = conv.sep
                intermediate_output = self.generate_sentence(image_tensor, stop_str, prompt)

                conv.messages[-1][-1] = intermediate_output
                conv.append_message(conv.roles[0], formatted_prompt[1])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                stop_str = conv.sep
                outputs = self.generate_sentence(image_tensor, stop_str, prompt)
            else:
                raise ValueError("len(prompt) must be 1 or 2")

        else:
            prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt
            
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep
            
            outputs = self.generate_sentence(image_tensor, stop_str, prompt)
        
        return outputs, intermediate_output
    
    def generate_sentence(self, image_tensor, stop_str, prompt):
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=0.0,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def generate_counterfactual_keywords(self, image, dataset_name):
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)
        factual_keywords, counterfactual_keywords = raw_sentence.split("Factual keywords:")[1].split("Counterfactual keywords:")
        factual_keywords = factual_keywords.replace('\n', '').strip()
        counterfactual_keywords = counterfactual_keywords.replace('\n', '').strip()
        return factual_keywords, counterfactual_keywords, prompt
    
    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass

    def get_coco_caption_prompt(self):
        raise NotImplementedError

        # def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None):
    #     image_path = os.path.join(self.temp_dir, "temp.jpg")
    #     raw_image_data.save(image_path)
        
    #     query = []
    #     query.append({"image": image_path})
    #     query.append({"text": 'Answer with only English. ' + text_prompt})
    #     query = self.tokenizer.from_list_format(query)
    #     response, history = self.model.chat(self.tokenizer, query=query, history=None)
        
    #     return response