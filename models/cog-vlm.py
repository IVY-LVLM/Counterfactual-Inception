import os
import io
import base64
import numpy as np
import torch

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image

from tools.read_yaml import get_hf_home
from .base_model import BaseModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

import warnings
warnings.filterwarnings("ignore")

default_path = "THUDM/cogvlm-chat-hf"

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

def get_device_map(model):
    device_map = infer_auto_device_map(model, max_memory={0:'16GiB',1:'20GiB',2:'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
    
    return device_map

class CogVLM(BaseModel):
    def __init__(self, temperature, max_new_tokens, model_name: str = "cog-vlm", model_path: str = default_path, counterfactual: bool = False,):
        super().__init__(model_name=model_name, model_path=model_path, counterfactual=counterfactual)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(default_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,)
        device_map = get_device_map(model)
        model = load_checkpoint_and_dispatch(model, os.path.join(get_hf_home(), 'hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c'), device_map=device_map,) 

        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = model.eval()
        self.generation_config = {"max_length": self.max_new_tokens, "do_sample": False, "temperature": self.temperature, "top_p": 0.0}

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, counterfactual_keyword = ""):
        intermediate_output = ""
        raw_image_data = get_pil_image(raw_image_data)
        raw_image_data = raw_image_data.convert("RGB")
        history = []
        
        if self.counterfactual:
            prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(prompt) == 1:
                query = prompt[0]
                response = self.generate_sentence(query, history, raw_image_data)
            elif len(prompt) == 2:
                query = prompt[0]
                intermediate_output = self.generate_sentence(query, history, raw_image_data)
                history.append((prompt[0], intermediate_output))
                query = prompt[1]
                response = self.generate_sentence(query, history, raw_image_data)
            else:
                raise ValueError("len(prompt) must be 1 or 2")
        else:
            query = text_prompt
            response = self.generate_sentence(query, history, raw_image_data)
        
        return response, intermediate_output
    
    def generate_sentence(self, query, history, image):
        gen_kwargs = {"max_length": self.max_new_tokens, "do_sample": False}

        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
    
            response = self.tokenizer.decode(outputs[0])
            outputs = response.split("</s>")[0]

        return outputs
    
    def generate_counterfactual_keywords(self, image, dataset_name):
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)[0]
        factual_keywords = raw_sentence.split('Factual Keywords: ')[-1].split('Counterfactual Keywords: ')[0].replace('\n', '')
        counterfactual_keywords = raw_sentence.split('Counterfactual Keywords: ')[-1].replace('\n', '')
        return factual_keywords, counterfactual_keywords, prompt
    
    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass

    def get_coco_caption_prompt(self):
        raise NotImplementedError