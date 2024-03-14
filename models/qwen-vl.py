import os
import io
import base64
import pickle
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image

from tools.read_yaml import get_data_folder
from .base_model import BaseModel
from accelerate import infer_auto_device_map
import warnings
warnings.filterwarnings("ignore")

default_path = "Qwen/Qwen-VL-Chat"
default_image_path = get_data_folder()

class QwenVL(BaseModel):
    def __init__(self, temperature, max_new_tokens, model_name: str = "qwen-vl", model_path: str = default_path, counterfactual: bool = False,):
        super().__init__(model_name=model_name, model_path=model_path, counterfactual=counterfactual)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        device_map = QwenVL.get_device_map()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model.generation_config.max_new_tokens = self.max_new_tokens
        self.model.generation_config.temperature = self.temperature
        self.model.generation_config.do_sample = False
        self.model.generation_config.top_p = 0

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, counterfactual_keyword = ""):
        intermediate_output = ""
        
        raw_image_data = os.path.join(default_image_path, dataset_name, image_path)
        
        if self.counterfactual:
            prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(prompt) == 1:
                query = []
                query.append({"image": raw_image_data})
                query.append({"text": 'Answer with only English. ' + prompt[0]})
                
                query = self.tokenizer.from_list_format(query)
                
                response, _ = self.model.chat(self.tokenizer, query=query, history=None)
                intermediate_output = ""
            elif len(prompt) == 2:
                query = []
                query.append({"image": raw_image_data})
                query.append({"text": 'Answer with only English. ' + prompt[0]})
                
                query = self.tokenizer.from_list_format(query)
                
                intermediate_output, history = self.model.chat(self.tokenizer, query=query, history=None)
                
                query = []
                query.append({"text": 'Answer with only English. ' + prompt[1]})
                
                query = self.tokenizer.from_list_format(query)
                
                response, _ = self.model.chat(self.tokenizer, query=query, history=history)
            else:
                raise ValueError("len(prompt) must be 1 or 2")
            
        else:
            query = []
            query.append({"image": raw_image_data})
            query.append({"text": 'Answer with only English. ' + text_prompt})
            query = self.tokenizer.from_list_format(query)
            
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        
        return response, intermediate_output
    
    def generate_counterfactual_keywords(self, image_file, dataset_name):
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        raw_sentence = self.generate(text_prompt=prompt, raw_image_data=None, dataset_name= dataset_name, image_path=os.path.basename(image_file))[0]
        factual_keywords = raw_sentence.split('Factual Keywords: ')[-1].split('Counterfactual Keywords: ')[0].replace('\n', '')
        counterfactual_keywords = raw_sentence.split('Counterfactual Keywords: ')[-1].replace('\n', '')
        return factual_keywords, counterfactual_keywords, prompt

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass

    def get_coco_caption_prompt(self):
        raise NotImplementedError

    @staticmethod
    def get_device_map():
        gpu_number = torch.cuda.device_count()
        
        if os.path.exists('.device_map')==False:
            os.mkdir('.device_map')

        filename = f"./.device_map/Qwen-VL_gpu{gpu_number}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                device_map = pickle.load(f)
                return device_map
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()
        max_memory = {}
        memory_per_gpu = str(int(30/gpu_number))
        for i in range(gpu_number):
            max_memory[i] = f"{memory_per_gpu}GiB"
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=['VisionTransformer', 'QWenBlock'])
        with open(filename, 'wb') as f:
            pickle.dump(device_map, f)
        del model
        return device_map
