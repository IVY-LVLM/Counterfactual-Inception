from turtle import end_fill
import math
from collections import defaultdict
import datetime
import json
import os
from enum import Enum

from tools.read_yaml import get_counterfactual_folder

default_counterfactuals_folder = get_counterfactual_folder()

'''
* Counterfactual file name format
{benchmark}_{model}_counterfactuals_{Month}{Day}_{version}.jsonl

* Counterfactual file content example
{"prompt": "Generate a counterfactual for the given image.", "temperature": 0.2, "max_new_tokens": 2048} 
{"id": "000000565045", "image": "000000565045.jpg", "keyword": "Curtains, Bathtub, Wooden Floor", "factual_keywords": "Towels, Shower, Tile Floor"}
{"id": "000000534041", "image": "000000534041.jpg", "keyword": "little girl, sandwich, soda bottle, dad feeding", "factual_keywords": "little boy, hot dog, water bottle, mom feeding"}
'''

class CounterFactualFileManager:
    def __init__(self, model, benchmark, contradiction_threshold, clip_lower_threshold, clip_upper_threshold, evaluate:bool = False) -> None:
        self.prompt = None
        self.temperature = None
        self.max_new_tokens = None

        self.contradiction_threshold = contradiction_threshold
        self.clip_lower_threshold = clip_lower_threshold
        self.clip_upper_threshold = clip_upper_threshold

        print()
        print('<<<<<=====CounterFactual file=====>>>>>')
        print("Here's the list of counterfactual files for the given model and benchmark: ")
        counterfactual_file_list = self._list_counterfactual_files(model, benchmark)
        for idx, file in enumerate(counterfactual_file_list):
            print("{}. {}".format(idx + 1, file))
        if evaluate:
            print("{}. {}".format(len(counterfactual_file_list) + 1, "Get counterfactual file of other model")) 
            print("===========================================================================")
            filename_idx = input("Enter the number of your selection({}-{}): ".format(1, len(counterfactual_file_list) + 1))
        else:
            print("{}. {}".format(len(counterfactual_file_list) + 1, "Create a new counterfactual file")) 
            print("{}. {}".format(len(counterfactual_file_list) + 2, "Get counterfactual file of other model")) 
            print("===========================================================================")
            filename_idx = input("Enter the number of your selection({}-{}): ".format(1, len(counterfactual_file_list) + 2))
        
        if evaluate:
            assert 1 <= int(filename_idx) <= len(counterfactual_file_list) + 1, "Invalid input"
            if int(filename_idx) == len(counterfactual_file_list) + 1:
                filename_idx = str(int(filename_idx) + 1)
        else:
            assert 1 <= int(filename_idx) <= len(counterfactual_file_list) + 2, "Invalid input"

        if int(filename_idx) == len(counterfactual_file_list) + 1:
            self.filename = self._get_new_counterfactual_filename(model, benchmark)  
            self.filepath = os.path.join(default_counterfactuals_folder, self.filename)
            from pathlib import Path
            Path(self.filepath).touch()
        elif int(filename_idx) == len(counterfactual_file_list) + 2:
            all_counterfactual_file_list = self._list_all_counterfactual_files(benchmark)
            for idx, file in enumerate(all_counterfactual_file_list):
                print("{}. {}".format(idx + 1, file))
            filename_idx = input("Enter the number of your selection({}-{}): ".format(1, len(all_counterfactual_file_list)))
            assert 1 <= int(filename_idx) <= len(all_counterfactual_file_list), "Invalid input"
            self.filename = all_counterfactual_file_list[int(filename_idx) - 1]
        else:
            self.filename = counterfactual_file_list[int(filename_idx) - 1]
        self.filepath = os.path.join(default_counterfactuals_folder, self.filename)
        print()
        print("Selected file: {}".format(self.filename))
        print("===========================================================================")
        print()
        print()
        self.cur_reviews = self.read_lines()
    
    def _find_files_with_pattern(self, directory, pattern):
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(pattern):
                    file_list.append(os.path.join(root, file))
        return file_list

    def _list_counterfactual_files(self, model, benchmark, date=None) -> list:
        if date == None:
            counterfactual_files = [f for f in os.listdir(default_counterfactuals_folder) if f.startswith("{}_{}_counterfactuals".format(benchmark, model))]
        else:
            counterfactual_files = [f for f in os.listdir(default_counterfactuals_folder) if f.startswith("{}_{}_counterfactuals_{}".format(benchmark, model, date))]
        return sorted(counterfactual_files)

    def _list_all_counterfactual_files(self, benchmark) -> list:
        counterfactual_files = [f for f in os.listdir(default_counterfactuals_folder) if f.startswith(benchmark) and f.endswith(".jsonl") and "counterfactuals" in f]
        return sorted(counterfactual_files)
    
    def _check_file_existence(self, filename:str) -> bool:
        return os.path.exists(os.path.join(default_counterfactuals_folder,filename)) and len(self.cur_reviews) != 0 

    def _get_new_counterfactual_filename(self, model, benchmark):
        date = "{}{}".format("%02d"%datetime.datetime.now().month, datetime.datetime.now().day)
        counterfactual_file_list = self._list_counterfactual_files(model, benchmark)
        new_filename = "{}_{}_counterfactuals_{}_v{}.jsonl".format(benchmark, model, date, len(counterfactual_file_list) + 1)
        print('Generate new file: ', new_filename)
        return new_filename

    def _check_settings(self, prompt, temperature, max_new_tokens):
        return prompt==self.prompt and temperature==self.temperature and max_new_tokens==self.max_new_tokens

    def save_counterfactual(self, image_file, counterfactual_keywords_with_score, prompt, temperature, max_new_tokens, subtask="", question=""):
        if not self._check_file_existence(self.filename):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": prompt, "temperature": temperature, "max_new_tokens": max_new_tokens}) + "\n")
            self.prompt = prompt
            self.temperature = temperature
            self.max_new_tokens = max_new_tokens
            self.cur_reviews = self.read_lines()
        elif self.prompt == None:
            self.prompt = self.cur_reviews[0]['prompt']
            self.temperature = self.cur_reviews[0]['temperature']
            self.max_new_tokens = self.cur_reviews[0]['max_new_tokens']
        
        if not self._check_settings(prompt, temperature, max_new_tokens):
            raise ValueError("Prompt, temperature, and max_new_tokens must be consistent. Please create a new counterfactual file.")
        
        result_dictionary = {}
        result_dictionary['id'] = os.path.basename(image_file).split('.')[0]
        result_dictionary["image"] = os.path.basename(image_file)
        result_dictionary["counterfactual_keywords_with_score"] = counterfactual_keywords_with_score
        result_dictionary["subtask"] = subtask
        result_dictionary["question"] = question

        with open(self.filepath, "a", encoding="utf-8") as f:
            json.dump(result_dictionary, f, ensure_ascii=False)
            f.write("\n") 
    
    def read_lines(self):
        if os.path.isfile(os.path.expanduser(self.filepath)):
            cur_reviews = [json.loads(line) for line in open(os.path.expanduser(self.filepath))]
        else:
            cur_reviews = []
        return cur_reviews

    def filter_conterfactuals(self, counterfactual_dictionary):
        counterfactual_filtered = []
        clip_score_list = []
        sorted_keyword = sorted(counterfactual_dictionary, key=lambda a: counterfactual_dictionary[a]['clip_score'][0])
        start_index = math.ceil(len(counterfactual_dictionary) * self.clip_lower_threshold)
        end_index = math.floor(len(counterfactual_dictionary) * self.clip_upper_threshold)
        sorted_keyword = sorted_keyword[start_index:end_index + 1]
        for keyword in sorted_keyword:
            if counterfactual_dictionary[keyword]['contradiction_score'][0] > self.contradiction_threshold:
                counterfactual_filtered.append(keyword)
                clip_score_list.append(counterfactual_dictionary[keyword]['clip_score'][0])
        return ', '.join(counterfactual_filtered)

    def get_counterfactuals(self):
        cf_file = os.path.join(self.filepath)
        counterfactuals_list = [json.loads(cf) for cf in open(os.path.expanduser(cf_file), "r")]
        if 'question' in counterfactuals_list[-1] and counterfactuals_list[-1]['question'] != "":
            counterfactuals = defaultdict(lambda: defaultdict(str))
            for counterfactual_item in counterfactuals_list[1:]:
                counterfactuals[counterfactual_item['id']][counterfactual_item['question']] = ', '.join(list(counterfactual_item['counterfactual_keywords_with_score'].keys()))
        else:
            counterfactuals = defaultdict(str)
            for counterfactual_item in counterfactuals_list[1:]:
                counterfactual_filtered = self.filter_conterfactuals(counterfactual_item['counterfactual_keywords_with_score'])
                counterfactuals[counterfactual_item['id']] = counterfactual_filtered
        return counterfactuals
    
    def is_absent_sample(self, idx):
        return idx + 1 >= len(self.cur_reviews) 