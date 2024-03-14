import datetime
import json
import os

from tools.read_yaml import get_log_folder

default_result_log_folder = get_log_folder()

'''
* Result file name format
{benchmark}_{model}_results_{Month}{Day}_{version}.jsonl

* Result file content
- firstline: counterfactual_file, temperature, max_new_tokens, clip_lower_threshold, clip_upper_threshold, contradiction_threshold, prompt_filename
- others: image_id, question, gt_ans, pred_ans, raw_pred, gpt_grade, category, counterfactual_keyword, intermediate_output

* Result file example
{"counterfactual_file": "mmvp_openai-gpt4_counterfactuals_0221_v1.jsonl", "temperature": 0.0, "max_new_tokens": 2048, "clip_lower_threshold": 0.2, "clip_upper_threshold": 0.8, "contradiction_threshold": 0.8, "prompt_filename": "240224-counterfactual-v3.txt"}
{"image_id": "1", "question": "Are the butterfly's wings closer to being open or closed? (a) Open (b) Closed", "gt_ans": "(a)", "pred_ans": "Open", "raw_pred": "Open", "gpt_grade": "yes", "category": null, "counterfactual_keyword": "distant shot, suburban garden, pet, cultivated garden, muted colors, feeding, domesticated insects, violet flowers, flying, macro shot, Moth, lavender flowers, Painted lady butterfly", "intermediate_output": ""}
{"image_id": "2", "question": "Are the butterfly's wings closer to being open or closed? (a) Open (b) Closed", "gt_ans": "(b)", "pred_ans": "Open", "raw_pred": "Open", "gpt_grade": "no", "category": null, "counterfactual_keyword": "wide shot, dragonfly, leaf, violet flowers, muted colors, flying, lavender flowers, moth, domestic pet", "intermediate_output": ""}
'''

class ResultFileManager:
    def __init__(self, model, benchmark, counterfactual) -> None:
        self.model = model
        self.benchmark = benchmark
        self.counterfactual = counterfactual
        self.default_result_log_folder = os.path.join(default_result_log_folder, benchmark.upper())
        self.default_result_log_folder = self.default_result_log_folder + "_CF" if counterfactual else self.default_result_log_folder
        if not os.path.exists(self.default_result_log_folder):
            os.mkdir(self.default_result_log_folder)
        self.filename = self._get_new_result_filename()
        print('<<<<<=====Result file=====>>>>>')
        print("Here's the list of result files for the given model and benchmark: ")
        print('1. New result file')
        result_files_list = self._list_result_files()
        for idx, file in enumerate(result_files_list):
            print("{}. {}".format(idx + 2, file))
        filename_idx = input(f"Enter the number of your selection(1-{len(result_files_list) + 1}): ")
        assert 1 <= int(filename_idx) <= len(result_files_list) + 1, "Invalid input"
        if int(filename_idx) == 1:
            self.filepath = os.path.join(self.default_result_log_folder, self.filename)
            self.cur_reviews = []
            print('New result file: ', self.filename)
            from pathlib import Path
            Path(self.filepath).touch()
        else:
            self.filename = result_files_list[int(filename_idx) - 2]
            self.filepath = os.path.join(self.default_result_log_folder, self.filename)
            self.cur_reviews = self._read_lines()
            print('Continue with the previous result file: ', self.filename)
        print("===========================================================================")
        print()
        print()

    def _list_result_files(self, date=None) -> list:
        if date == None:
            result_files = [f for f in os.listdir(self.default_result_log_folder) if f.startswith("{}_{}_results".format(self.benchmark, self.model))]
        else:
            result_files = [f for f in os.listdir(self.default_result_log_folder) if f.startswith("{}_{}_results_{}".format(self.benchmark, self.model, date))]
        return sorted(result_files)

    def _get_new_result_filename(self):
        date = "{}{}".format("%02d"%datetime.datetime.now().month, datetime.datetime.now().day)
        result_files_list = self._list_result_files()
        version_list = []
        for filename in result_files_list:
            if date in filename:
                version_list.append(int(filename.split("_v")[-1].split('_')[0].split(".")[0]))
        version = max(version_list) + 1 if version_list else 1
        if self.counterfactual:
            new_filename = "{}_{}_results_{}_v{}_cf.jsonl".format(self.benchmark, self.model, date, version)
        else:
            new_filename = "{}_{}_results_{}_v{}.jsonl".format(self.benchmark, self.model, date, version)
        return new_filename

    def _check_file_existence(self, filename:str) -> bool:
        return os.path.exists(os.path.join(self.default_result_log_folder, filename)) and len(self.cur_reviews) != 0 
    
    def get_temperature_max_new_tokens(self, model):
        if hasattr(model, "temperature"):
            temperature = model.temperature
        else:
            temperature = -1.0
        if hasattr(model, "max_new_tokens"):
            max_new_tokens = model.max_new_tokens
        else:
            max_new_tokens = -1
        return temperature, max_new_tokens

    def save_result(self, image_id, question, gt_ans, pred_ans, raw_pred, counterfactual_file, temperature, max_new_tokens, mmvp_gpt_grade="", category_pope_mme="", counterfactual_keyword="", intermediate_output="", clip_lower_threshold=-1.0, clip_upper_threshold=-1.0, contradiction_threshold=-1.0, prompt_filename = ""):
        if not self._check_file_existence(self.filename):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"counterfactual_file": counterfactual_file, "temperature": temperature, "max_new_tokens": max_new_tokens, "clip_lower_threshold": clip_lower_threshold, "clip_upper_threshold": clip_upper_threshold,"contradiction_threshold": contradiction_threshold, "prompt_filename": prompt_filename}) + "\n")
            self.cur_reviews = self._read_lines()
        else:
            assert self.cur_reviews[0]["counterfactual_file"] == counterfactual_file, "counterfactual_file is different from the previous one"
            assert self.cur_reviews[0]["temperature"] == temperature, "temperature is different from the previous one"
            assert self.cur_reviews[0]["max_new_tokens"] == max_new_tokens, "max_new_tokens is different from the previous one"
            if "prompt_filename" in self.cur_reviews[0]:
                assert self.cur_reviews[0]["prompt_filename"] == prompt_filename, "prompt_filename is different from the previous one"
            if "clip_lower_threshold" in self.cur_reviews[0]:
                assert self.cur_reviews[0]["clip_lower_threshold"] == clip_lower_threshold, "clip_lower_threshold is different from the previous one"
            if "clip_upper_threshold" in self.cur_reviews[0]:
                assert self.cur_reviews[0]["clip_upper_threshold"] == clip_upper_threshold, "clip_upper_threshold is different from the previous one"
            if "contradiction_threshold" in self.cur_reviews[0]:
                assert self.cur_reviews[0]["contradiction_threshold"] == contradiction_threshold, "contradiction_threshold is different from the previous one"

        result_dict = {}
        result_dict['image_id'] = image_id
        result_dict['question'] = question
        result_dict['gt_ans'] = gt_ans
        result_dict['pred_ans'] = pred_ans
        result_dict['raw_pred'] = raw_pred
        result_dict['gpt_grade'] = mmvp_gpt_grade
        result_dict['category'] = category_pope_mme
        result_dict['counterfactual_keyword'] = counterfactual_keyword
        result_dict['intermediate_output'] = intermediate_output
        with open(self.filepath, "a", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False)
            f.write("\n") 
    
    def _read_lines(self):
        if os.path.isfile(os.path.expanduser(self.filepath)):
            cur_reviews = [json.loads(line) for line in open(os.path.expanduser(self.filepath))]
        else:
            cur_reviews = []
        return cur_reviews
    
    def is_absent_sample(self, idx):
        return idx + 1 >= len(self.cur_reviews) 
    
    def get_results(self, row, item_name):
        return self.cur_reviews[row + 1][item_name]
    
    def save_evaluation(self, evaluation):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(evaluation) + "\n")