import os, sys
import json
import math
import time
import numpy as np
import openai
import pytz, datetime
import requests

from tqdm import tqdm
from collections import defaultdict

from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from PIL import Image

NUM_SECONDS_TO_SLEEP = 0.5

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)
result_dict = defaultdict(lambda: defaultdict(lambda: ""))

context_file = "benchmarks/llavabench/context.jsonl"
compare_gpt4_file = "benchmarks/llavabench/answers_gpt4.jsonl"
rule_file = "benchmarks/llavabench/rule.json"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

api_key = get_openai_api_key()
openai.api_key = api_key
def get_eval(content: str, max_tokens: int):
    model = "gpt-4-0125-preview" #"gpt-4-0613"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    messages = [
            {
                'role': 'system',
                'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
            }, {
                'role': 'user',
                'content': content,
            }
            ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }
    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            outputs = response.json()['choices'][0]['message'].get('content')
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return outputs


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]

class LLaVABenchDataset(BaseEvalDataset):
    def __init__(self, data_path: str = "llavabench", default_output_path="LLaVA-Bench", counterfactual = False, counterfactual_path = "",contradiction_threshold=-1.0, clip_lower_threshold=-1.0, clip_upper_threshold=-1.0):
        super().__init__("llavabench", data_path, counterfactual=counterfactual, counterfactual_path = counterfactual_path, contradiction_threshold=contradiction_threshold, clip_lower_threshold=clip_lower_threshold, clip_upper_threshold=clip_upper_threshold)
        default_output_path = os.path.join(get_log_folder(), default_output_path)
        if self.counterfactual:
            default_output_path = default_output_path + "_CF"
        self.data_path = os.path.join(get_data_folder(), data_path)
        self.default_output_path = default_output_path

    def _evaluate(self, model, counterfactual_file_manager, result_file_manager):
        super()._evaluate(model, counterfactual_file_manager, result_file_manager)
        
        # Question
        question_file = "benchmarks/llavabench/questions.jsonl"
        image_folder = self.data_path

        num_chunks = 1
        chunk_idx = 0
        
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        questions = get_chunk(questions, num_chunks, chunk_idx)

        # context
        context_list = [json.loads(line) for line in open(os.path.expanduser(context_file), "r")]
        image_to_context = {context['image']: context for context in context_list}

        # compare_gpt4
        # gpt4_answers = [json.loads(q) for q in open(os.path.expanduser(compare_gpt4_file), "r")]
        gpt4_answers = [json.loads(line) for line in open(os.path.expanduser(compare_gpt4_file))]
        gpt4_answers = {q['question_id']: q for q in gpt4_answers}

        # rule
        rule_dict = json.load(open(os.path.expanduser(rule_file), 'r'))

        if self.counterfactual:
            counterfactuals = self.counterfactual_file_manager.get_counterfactuals()

        total_score = defaultdict(list)
        for row_idx, line in tqdm(enumerate(questions)):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            category = line["category"]
                        
            question = qs
            
            image_id = image_file.split(".")[0]
            
            if self.result_file_manager.is_absent_sample(row_idx) == False:
                scores = self.result_file_manager.get_results(row_idx, 'pred_ans')
                print('skipping: ', row_idx, image_id)
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.counterfactual:
                    try:
                        counterfactual_keyword = counterfactuals[image_id][question]
                    except:
                        counterfactual_keyword = counterfactuals[image_id]
                    if 'detail' in category: 
                        response, intermediate_output = model.generate(question, image, dataset_name="llavabench", image_path = image_file, counterfactual_keyword = counterfactual_keyword)
                    else:
                        response = ""
                        intermediate_output = ""
                else:
                    if 'detail' in category:
                        response, intermediate_output = model.generate(question, image, dataset_name="llavabench", image_path = image_file)
                    else:
                        response = ""
                        intermediate_output = ""
                gpt_ans = gpt4_answers[idx]

                inst = image_to_context[image_file]

                if isinstance(inst['caption'], list):
                    cap_str = '\n'.join(inst['caption'])
                else:
                    cap_str = inst['caption']

                category = 'llava_bench_' + category
                if category in rule_dict:
                    rule = rule_dict[category]
                else:
                    assert False, f"Visual QA category not found in rule file: {category}."
                prompt = rule['prompt']
                role = rule['role']
                content = (f'[Context]\n{cap_str}\n\n'
                        f'[Question]\n{question}\n\n'
                        f'[{role} 1]\n{gpt_ans["text"]}\n\n[End of {role} 1]\n\n'
                        f'[{role} 2]\n{response}\n\n[End of {role} 2]\n\n'
                        f'[System]\n{prompt}\n\n')

                max_tokens = 1024
                review = get_eval(content, max_tokens)
                scores = parse_score(review)

                temperature, max_new_tokens = self.result_file_manager.get_temperature_max_new_tokens(model)
                counterfactual_file = self.counterfactual_file_manager.filename if hasattr(self, "counterfactual_file_manager") else ""
                self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans="", pred_ans=scores, raw_pred=response, counterfactual_file=counterfactual_file, temperature=temperature, max_new_tokens=max_new_tokens, mmvp_gpt_grade=None, category_pope_mme=category, counterfactual_keyword=counterfactual_keyword, intermediate_output=intermediate_output, clip_lower_threshold=self.clip_lower_threshold, clip_upper_threshold=self.clip_upper_threshold, contradiction_threshold=self.contradiction_threshold, prompt_filename=self.prompt_filename)

            total_score[category].append(scores)
            total_score['all'].append(scores)

        result_dict = defaultdict(list)
        for k, v in sorted(total_score.items()):
            stats = np.asarray(v).mean(0).tolist()
            result_dict[k] = [round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1)]
            print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
            
        self.result_file_manager.save_evaluation(total_score)
        self.result_file_manager.save_evaluation(result_dict)
            
    def generate_counterfactual_regarding_question(self, model, counterfactual_file_manager, constant_prompt_format):
        
        question_file = os.path.join(self.data_path, "questions.jsonl")
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        num_chunks = 1
        chunk_idx = 0
        questions = get_chunk(questions, num_chunks, chunk_idx)
        image_folder = os.path.join(self.data_path, "images")
        row_idx = 0
        
        for row_idx, line in tqdm(enumerate(questions)):
            image_file = line["image"]
            image_id = image_file.split(".")[0]
            if counterfactual_file_manager.is_absent_sample(row_idx):
                question = line["text"]
                category = line["category"]
                prompt = constant_prompt_format.format(question)

                image_path = os.path.join(self.data_path, image_file)
                image_file = os.path.basename(image_path)
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                counterfactual_keywords_dictionary = defaultdict(lambda: defaultdict(list))

                counterfactual_keywords, _ = model.generate(prompt, image, 'llavabench', image_path = image_path)
                counterfactual_keywords_line = counterfactual_keywords.split('\n')
                counterfactual_keywords = [item.split('.')[-1].strip() for item in counterfactual_keywords_line if '1.' in item or '2.' in item or '3.' in item]
                for counterfactual_word in counterfactual_keywords:
                    counterfactual_keywords_dictionary[counterfactual_word]['factual_word'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['contradiction_score'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['clip_score'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['turn_number'] = [""]
                counterfactual_file_manager.save_counterfactual(image_file=image_file, counterfactual_keywords_with_score=counterfactual_keywords_dictionary, prompt=constant_prompt_format, temperature=model.temperature, max_new_tokens=model.max_new_tokens, subtask="", question=question)
            else:
                print('skipping: ', row_idx, image_id)
            

    
    
