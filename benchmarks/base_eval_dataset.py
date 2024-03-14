from abc import ABC, abstractmethod
import os
import sys
from PIL import Image
from typing import Dict, List, Any

import importlib

from tqdm import tqdm

AVAILABLE_EVAL_DATASETS: Dict[str, str] = {
    "mmvp": "MMVPDataset",
    "pope": "PopeDataset",
    "llavabench": "LLaVABenchDataset",
    "llava-qa90": "LLaVAQA90Dataset",
    "mmhalbench": "MMHalBenchDataset",
}


class BaseEvalDataset(ABC):
    def __init__(self, name: str, dataset_path: str, *, max_batch_size: int = 1, counterfactual: bool = False, counterfactual_path: str = "", contradiction_threshold: float = 0.5, clip_lower_threshold: float = 0.5, clip_upper_threshold: float = 0.5):
        self.name = name
        self.dataset_path = dataset_path
        self.max_batch_size = max_batch_size
        self.counterfactual = counterfactual
        self.counterfactual_path = counterfactual_path
        self.contradiction_threshold = contradiction_threshold
        self.clip_lower_threshold = clip_lower_threshold
        self.clip_upper_threshold = clip_upper_threshold

    def evaluate(self, model, counterfactual_file_manager, result_file_manager):
        return self._evaluate(model, counterfactual_file_manager, result_file_manager)

    @abstractmethod
    def _evaluate(self, model: str, counterfactual_file_manager, result_file_manager):
        if hasattr(model, "counterfactual_prompt_manager"):
            self.prompt_filename = model.counterfactual_prompt_manager.filename
        else:
            self.prompt_filename = ""
        if self.counterfactual:
            self.counterfactual_file_manager = counterfactual_file_manager
        self.result_file_manager = result_file_manager


def load_dataset(dataset_name: str, dataset_args: Dict[str, str] = {}) -> BaseEvalDataset:
    assert dataset_name in AVAILABLE_EVAL_DATASETS, f"{dataset_name} is not an available eval dataset."
    module_path = "benchmarks." + dataset_name
    dataset_formal_name = AVAILABLE_EVAL_DATASETS[dataset_name]
    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, dataset_formal_name)
    print(f"Imported class: {dataset_class}")
    # import pdb;pdb.set_trace()
    # get dataset args without "name"
    init_args = dataset_args.copy()
    init_args.pop("name")
    return dataset_class(**init_args)
