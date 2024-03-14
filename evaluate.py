import sys
import argparse
import os

import yaml
import contextlib

from tools.read_yaml import *

sys.path.append(os.getcwd())

from benchmarks.base_eval_dataset import load_dataset
from file_utils.counterfactual_file_manage import CounterFactualFileManager
from file_utils.result_file_manage import ResultFileManager
from models.base_model import AVAILABLE_MODELS, load_model
from transformers import set_seed

set_seed(555)

def get_info(info):
    if "name" not in info:
        raise ValueError("Model name is not specified.")
    name = info["name"]
    # info.pop("name")
    return name, info


def load_models(model_infos):
    for model_info in model_infos:
        name, info = get_info(model_info)
        model = load_model(name, info)
        yield model


def load_datasets(dataset_infos):
    for dataset_info in dataset_infos:
        name, info = get_info(dataset_info)
        dataset = load_dataset(name, info)
        yield dataset


class DualOutput:
    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--models",
        type=str,
        nargs="?",
        help="Specify model names as comma separated values.",
        default='llama_adapter,uniter,uniter_large',
    )
    args.add_argument(
        "--datasets",
        type=str,
        nargs="?",
        help="Specify dataset names as comma separated values.",
        default='vqa2,mscoco',
    )
    args.add_argument(
        "--counterfactual",
        action="store_true",
        default=False,
    )
    args.add_argument(
        "--contradiction_threshold",
        type=float,
        default=0.8,
    )
    args.add_argument(
        "--clip_lower_threshold",
        type=float,
        default=0.2,
    )
    args.add_argument(
        "--clip_upper_threshold",
        type=float,
        default=0.8,
    )
    args.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
    )
    args.add_argument( 
        "--temperature",
        type=float,
        default=0.0,
    )

    phrased_args = args.parse_args()

    model_names = phrased_args.models.split(",")
    model_infos = [{"name": name, "counterfactual": phrased_args.counterfactual, "temperature": phrased_args.temperature, "max_new_tokens": phrased_args.max_new_tokens} for name in model_names]
    dataset_infos = [{"name": dataset_name, "counterfactual": phrased_args.counterfactual, "counterfactual_path": get_counterfactual_folder(), "contradiction_threshold": phrased_args.contradiction_threshold, "clip_lower_threshold": phrased_args.clip_lower_threshold, "clip_upper_threshold": phrased_args.clip_upper_threshold} for dataset_name in phrased_args.datasets.split(",")]

    if not os.path.exists(get_log_folder()):
        os.makedirs(get_log_folder())
    
    if not os.path.exists(get_counterfactual_folder()):
        os.makedirs(get_counterfactual_folder())
    
    is_counterfactual = phrased_args.counterfactual

    for model_info in model_infos:
        print("\nMODEL INFO:", model_info)
        print("-" * 80)
        dataset_count = 0
        for dataset_info in dataset_infos:
            dataset_name, _dataset_info = get_info(dataset_info)
            
            counterfactual_file_manager = CounterFactualFileManager(model_info["name"], dataset_name, contradiction_threshold=phrased_args.contradiction_threshold, clip_lower_threshold=phrased_args.clip_lower_threshold, clip_upper_threshold=phrased_args.clip_upper_threshold, evaluate=True) if is_counterfactual else None
            result_file_manager = ResultFileManager(model_info["name"], dataset_name, is_counterfactual)

            model = load_model(model_info["name"], model_info)
            dataset = load_dataset(dataset_name, _dataset_info)

            dataset_count += 1
            print('MODEL:', model.name, 'TEMPERATURE:', model.temperature, 'MAX_NEW_TOKENS:', model.max_new_tokens, 'CONTRADICTION:', dataset.contradiction_threshold, 'CLIP:', dataset.clip_lower_threshold, dataset.clip_upper_threshold)
            print(f"\nDATASET: {dataset.name}")
            print("-" * 20)

            dataset.evaluate(model, counterfactual_file_manager, result_file_manager)  # Assuming this function now prints results directly.
            print()

        print("-" * 80)
        print(f"Total Datasets Evaluated: {dataset_count}\n")

    print("=" * 80)

# python evaluate.py --models otter_image --datasets mmbench
