from collections import defaultdict
import os
import datetime
from tqdm import tqdm, trange

from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
from typing import Union


class PopeDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path="Otter-AI/POPE",
        split="test",
        default_output_path="POPE",
        batch_size=1,
        counterfactual = False,
        counterfactual_path = "",
        contradiction_threshold=-1.0, 
        clip_lower_threshold=-1.0,
        clip_upper_threshold=-1.0
    ):
        super().__init__("pope", data_path, counterfactual=counterfactual, counterfactual_path = counterfactual_path, contradiction_threshold=contradiction_threshold, clip_lower_threshold=clip_lower_threshold, clip_upper_threshold=clip_upper_threshold) #, max_batch_size=batch_size)
        default_output_path = os.path.join(get_log_folder(), default_output_path)
        if self.counterfactual:
            default_output_path = default_output_path + "_CF"

        print("Loading dataset from", data_path)
        self.data = load_dataset(data_path, split=split)
        print("Dataset loaded")
        self.default_output_path = default_output_path
        if not os.path.exists(default_output_path):
            os.makedirs(default_output_path)

    def parse_pred(self, text):
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "").lower()
        words = text.split(" ")

        if "not" in words or "no" in words:
            return "no"
        else:
            return "yes"

    def _evaluate(self, model, counterfactual_file_manager, result_file_manager):
        super()._evaluate(model, counterfactual_file_manager, result_file_manager)

        metrics = {
            "adversarial": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "popular": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "random": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "overall": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
        }

        if self.counterfactual:
            counterfactuals = self.counterfactual_file_manager.get_counterfactuals()

        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for row_idx, pope_data in enumerate(self.data):
                question = pope_data["question"]
                gt_ans = pope_data["answer"]
                image = pope_data["image"]
                image_id = pope_data["image_source"]
                category = pope_data["category"]

                if self.result_file_manager.is_absent_sample(row_idx) == False:
                    pred_ans = self.result_file_manager.get_results(row_idx, 'pred_ans')
                    image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                    print('skipping: ', row_idx, image_id)
                else:
                    if self.counterfactual:
                        counterfactual_keyword = counterfactuals[image_id]
                        if category == "adversarial":
                            response, intermediate_output = model.generate(question, image, 'pope', image_path = image_id + '.jpg', counterfactual_keyword = counterfactual_keyword)
                        else:
                            response = ""
                            intermediate_output = ""
                    else:
                        counterfactual_keyword = ""
                        if category == "adversarial":
                            response, intermediate_output = model.generate(question, image, 'pope', image_path = image_id + '.jpg')
                        else:
                            response = ""
                            intermediate_output = ""

                    pred_ans = self.parse_pred(response)

                    temperature, max_new_tokens = self.result_file_manager.get_temperature_max_new_tokens(model)
                    counterfactual_file = self.counterfactual_file_manager.filename if hasattr(self, "counterfactual_file_manager") else ""
                    self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans=gt_ans, pred_ans=pred_ans, raw_pred=response, counterfactual_file=counterfactual_file, temperature=temperature, max_new_tokens=max_new_tokens, mmvp_gpt_grade=None, category_pope_mme=category, counterfactual_keyword=counterfactual_keyword, intermediate_output=intermediate_output, clip_lower_threshold=self.clip_lower_threshold, clip_upper_threshold=self.clip_upper_threshold, contradiction_threshold=self.contradiction_threshold, prompt_filename=self.prompt_filename)

                answer = gt_ans

                if pred_ans == "yes":
                    metrics[category]["yes_count"] += 1
                    metrics["overall"]["yes_count"] += 1
                else:
                    metrics[category]["no_count"] += 1
                    metrics["overall"]["no_count"] += 1

                if pred_ans == answer and pred_ans == "yes":
                    metrics[category]["TP"] += 1
                    metrics["overall"]["TP"] += 1
                elif pred_ans == answer and pred_ans == "no":
                    metrics[category]["TN"] += 1
                    metrics["overall"]["TN"] += 1
                elif pred_ans != answer and pred_ans == "yes":
                    metrics[category]["FP"] += 1
                    metrics["overall"]["FP"] += 1
                else:
                    metrics[category]["FN"] += 1
                    metrics["overall"]["FN"] += 1

                pbar.update(1)

        for category in metrics:
            print(f"----------- {category} -----------")

            TP = metrics[category]["TP"]
            TN = metrics[category]["TN"]
            FP = metrics[category]["FP"]
            FN = metrics[category]["FN"]
            yes_count = metrics[category]["yes_count"]
            no_count = metrics[category]["no_count"]

            print("TP\tFP\tTN\tFN\t")
            print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

            if TP + FP == 0:
                metrics[category]["precision"] = precision = 0
            else:
                metrics[category]["precision"] = precision = float(TP) / float(TP + FP)

            if TP + FN == 0:
                metrics[category]["recall"] = recall = 0
            else:
                metrics[category]["recall"] = recall = float(TP) / float(TP + FN)

            if precision + recall == 0:
                metrics[category]["f1"] = f1 = 0
            else:
                metrics[category]["f1"] = f1 = 2 * precision * recall / float(precision + recall)

            metrics[category]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)

            if yes_count + no_count == 0:
                metrics[category]["yes_ratio"] = yes_ratio = 0
            else:
                metrics[category]["yes_ratio"] = yes_ratio = yes_count / float(yes_count + no_count)

            print("Accuracy: {}".format(acc))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1 score: {}".format(f1))
            print("Yes ratio: {}".format(yes_ratio))

            result_dict = {}
            result_dict[category] = {} 
            result_dict[category]['acc'] = acc
            result_dict[category]['precision'] = precision
            result_dict[category]['recall'] = recall
            result_dict[category]['f1'] = f1
            result_dict[category]['yes_ratio'] = yes_ratio
            self.result_file_manager.save_evaluation(result_dict)
            

        print(f"----------- overall -----------")

        TP = metrics["overall"]["TP"]
        TN = metrics["overall"]["TN"]
        FP = metrics["overall"]["FP"]
        FN = metrics["overall"]["FN"]
        yes_count = metrics["overall"]["yes_count"]
        no_count = metrics["overall"]["no_count"]

        print("TP\tFP\tTN\tFN\t")
        print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

        metrics["overall"]["precision"] = precision = float(TP) / float(TP + FP)
        metrics["overall"]["recall"] = recall = float(TP) / float(TP + FN)
        metrics["overall"]["f1"] = f1 = 2 * precision * recall / float(precision + recall)
        metrics["overall"]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)
        metrics["overall"]["yes_ratio"] = yes_ratio = float(yes_count) / float(yes_count + no_count)

        print("Accuracy: {}".format(acc))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 score: {}".format(f1))
        print("Yes ratio: {}".format(yes_ratio))

        result_dict = {}
        result_dict['overall'] = {} 
        result_dict['overall']['acc'] = acc
        result_dict['overall']['precision'] = precision
        result_dict['overall']['recall'] = recall
        result_dict['overall']['f1'] = f1
        result_dict['overall']['yes_ratio'] = yes_ratio
        self.result_file_manager.save_evaluation(result_dict)

        return metrics

    def generate_counterfactual_regarding_question(self, model, counterfactual_file_manager, constant_prompt_format):
        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for row_idx, pope_data in enumerate(self.data):
                image_id = pope_data["image_source"]
                if self.result_file_manager.is_absent_sample(row_idx) == False:
                    print('skipping: ', row_idx, image_id)
                else:
                    question = pope_data["question"]
                    prompt = constant_prompt_format.format(question)
                    image = pope_data["image"]
                    category = pope_data["category"]
                    image_file =  image_id + '.jpg'
                    counterfactual_keywords, _ = model.generate(prompt, image, 'pope', image_path = image_id + '.jpg')
                    counterfactual_keywords_line = counterfactual_keywords.split('\n')
                    counterfactual_keywords = [item.split('.')[-1].strip() for item in counterfactual_keywords_line if '1.' in item or '2.' in item or '3.' in item]
                    counterfactual_keywords_dictionary = defaultdict(lambda: defaultdict(list))
                    for counterfactual_word in counterfactual_keywords:
                        counterfactual_keywords_dictionary[counterfactual_word]['factual_word'] = [""]
                        counterfactual_keywords_dictionary[counterfactual_word]['contradiction_score'] = [""]
                        counterfactual_keywords_dictionary[counterfactual_word]['clip_score'] = [""]
                        counterfactual_keywords_dictionary[counterfactual_word]['turn_number'] = [""]
                    counterfactual_file_manager.save_counterfactual(image_file=image_file, counterfactual_keywords_with_score=counterfactual_keywords_dictionary, prompt=constant_prompt_format, temperature=model.temperature, max_new_tokens=model.max_new_tokens, subtask=category, question=question)