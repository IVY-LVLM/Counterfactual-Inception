import pathlib
import sys

cur_path = sys.path[0]
sys.path.append("../..")
wp_cur_dir = pathlib.Path(cur_path)
sys.path.append(cur_path)
sys.path.append(str(wp_cur_dir.parent))
sys.path.append(str(wp_cur_dir.parent.parent))
print(sys.path)

from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib
from file_utils.counterfactual_utilization_prompt_manage import CFUtilizationPromptManager

AVAILABLE_MODELS: Dict[str, str] = {
    "qwen-vl": "QwenVL",
    "llava-model-hf": "LLaVA_Model_HF",
    "gpt4v": "OpenAIGPT4Vision",
    "gemini": "Gemini",
    "cog-vlm": "CogVLM",
    "yi-vl": "YiVL",
}


class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str, *, max_batch_size: int = 1, counterfactual: bool = False):
        self.name = model_name
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.counterfactual = counterfactual
        if self.counterfactual:
            self.counterfactual_prompt_manager = CFUtilizationPromptManager()

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def eval_forward(self, **kwargs):
        pass
    
    @abstractmethod
    def get_coco_caption_prompt(self):
        pass

def load_model(model_name: str, model_args: Dict[str, str]) -> BaseModel:
    assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
    module_path = "models." + model_name
    model_formal_name = AVAILABLE_MODELS[model_name]
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_formal_name)
    print(f"Imported class: {model_class}")
    model_args.pop("name")
    return model_class(**model_args)
