import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

def load_pretrained_model(model_path, model_base, model_name, load_8bit=True, load_4bit=True, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
        # kwargs['no_split_module_classes']=['FalconDecoderLayer']
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
        
    model = LlavaForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    # model = LlavaForConditionalGeneration.from_pretrained(model_path)
    # model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
    # model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, **kwargs) # , device_map=device_map) #, load_in_8bit=load_8bit)
    # processor = AutoProcessor.from_pretrained(model_path)
    context_len = 2048

    return model, processor, context_len
