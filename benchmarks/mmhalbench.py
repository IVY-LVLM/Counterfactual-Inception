from datasets import load_dataset
from collections import defaultdict
from email.policy import default
import json
import numpy as np
from tqdm import tqdm

from tools.read_yaml import *
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
import os
import datetime
from PIL import Image  
import PIL  
import pandas as pd
from openai import OpenAI
import time
import re
import pytz
from typing import Union
api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)
utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.
For clarity, consider these examples:
### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.
### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.
### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.
### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.
### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.
With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.
Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.
To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination
### Image Contents
{}
### Question
{}
### Standard Human-Generated Answer
{}
### LMM Response to Evaluate
{}
'''

def chatgpt_grade_response(image_content, question, gt_answer, model_answer):
    input_text = template.format(image_content, question, gt_answer, model_answer)
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model= 'gpt-4-0125-preview', #'gpt-4-0613',#
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0, 
            )
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue
    return response.choices[0].message.content

class MMHalBenchDataset(BaseEvalDataset):
    def __init__(self, data_path: str = "MMHALBENCH", split="test", default_output_path='MMHALBENCH', counterfactual = False,
        counterfactual_path = "", contradiction_threshold=-1.0, clip_lower_threshold=-1.0, clip_upper_threshold=1.0):
        super().__init__("mmhalbench", data_path, counterfactual=counterfactual, counterfactual_path = counterfactual_path, contradiction_threshold=contradiction_threshold, clip_lower_threshold=clip_lower_threshold, clip_upper_threshold=clip_upper_threshold)
        
        self.default_output_path = os.path.join(get_log_folder(), default_output_path)
        if self.counterfactual:
            self.default_output_path = self.default_output_path + "_CF"

    def _evaluate(self, model, counterfactual_file_manager, result_file_manager):
        super()._evaluate(model, counterfactual_file_manager, result_file_manager)

        dataset = load_dataset("Shengcao1006/MMHal-Bench")

        counterfactuals = []
        scores = []
        hallucination_list = []
        scores_each = [[] for _ in range(8)]
        for row_idx, data in tqdm(enumerate(dataset['test'])):
            id = data['id']
            question_type = data['question_type']
            question_topic = data['question_topic']
            # image_id = data['image_id']
            image_id = os.path.basename(data['image_path']).replace(".jpg", "")
            image_src = data['image_src']
            image_content = data['image_content']
            question = data['question']
            gt_answer = data['gt_answer']
            model_answer = data['model_answer']
            image_path = data['image_path']
            if self.result_file_manager.is_absent_sample(row_idx) == False:
                pred_ans = self.result_file_manager.get_results(row_idx, 'pred_ans')
                image_id = self.result_file_manager.get_results(row_idx, 'image_id')
                gpt_grade = self.result_file_manager.get_results(row_idx, 'gpt_grade')
                hallucination = 1 if pred_ans == "hallucination" else 0
                print('skipping: ', row_idx, image_id)
            else:
                if self.counterfactual:
                    counterfactuals = self.counterfactual_file_manager.get_counterfactuals()
                cur_prompt = question

                image = Image.open(image_path).convert('RGB')
                if self.counterfactual:
                    counterfactual_keyword = counterfactuals[image_id]
                    response, intermediate_output = model.generate(cur_prompt, image, 'mmvp', image_path = image_path, counterfactual_keyword = counterfactual_keyword)
                else:
                    counterfactual_keyword = ""
                    response, intermediate_output = model.generate(cur_prompt, image, 'mmvp', image_path = image_path) 
                question, correct_answer, raw_pred = cur_prompt, gt_answer, response

                gpt4_response = chatgpt_grade_response(image_content, question, correct_answer, raw_pred)
                gpt_grade = -1
                scores_found = []
                for s in range(7):
                    if f'rating: {s}' in gpt4_response.lower():
                        scores_found.append(s)
                if len(scores_found) == 1:
                    gpt_grade = scores_found[0]
                else:
                    print('Warning: multiple or zero scores found')
                    print(row_idx, response)
                    gpt_grade = 0
                
                if gpt_grade >= 3:
                    hallucination = 0
                    pred_ans = "no hallucination"
                else:
                    hallucination = 1
                    pred_ans = "hallucination"
                
                temperature, max_new_tokens = self.result_file_manager.get_temperature_max_new_tokens(model)
                counterfactual_file = self.counterfactual_file_manager.filename if hasattr(self, "counterfactual_file_manager") else ""
                self.result_file_manager.save_result(image_id=image_id, question=question, gt_ans=correct_answer, pred_ans=pred_ans, raw_pred=raw_pred, counterfactual_file=counterfactual_file, temperature=temperature, max_new_tokens=max_new_tokens, mmvp_gpt_grade=gpt_grade, category_pope_mme=question_type, counterfactual_keyword=counterfactual_keyword, intermediate_output=intermediate_output, clip_lower_threshold=self.clip_lower_threshold, clip_upper_threshold=self.clip_upper_threshold, contradiction_threshold=self.contradiction_threshold, prompt_filename=self.prompt_filename)
            
            scores.append(gpt_grade)
            hallucination_list.append(hallucination)
            
            question_type = row_idx % 8
            scores_each[question_type].append(gpt_grade)
        
        average_score = sum(scores) / len(scores)
        hallucination_rate = sum(hallucination_list) / len(hallucination_list)
        average_score_each = [str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]

        result_dict = defaultdict(str)
        result_dict['average_score'] = str(average_score)
        result_dict['hallucination_rate'] = str(hallucination_rate)
        result_dict['average_score_each'] = ','.join(average_score_each)

        self.result_file_manager.save_evaluation(result_dict)

        print('Average score: {:.2f}'.format(average_score))
        print('Hallucination rate: {:.2f}'.format(hallucination_rate))
        print('Average score for each question type:', ','.join(average_score_each), flush=True)

    def generate_counterfactual_regarding_question(self, model, counterfactual_file_manager, constant_prompt_format):
        row_idx = 0
        dataset = load_dataset("Shengcao1006/MMHal-Bench")

        for row_idx, data in tqdm(enumerate(dataset['test'])):
            image_id = os.path.basename(data['image_path']).replace(".jpg", "")
            if self.counterfactual_file_manager.is_absent_sample(row_idx) == False:
                print('skipping: ', row_idx, image_id)
            else:
                id = data['id']
                image_src = data['image_src']
                image_content = data['image_content']
                question = data['question']
                gt_answer = data['gt_answer']
                model_answer = data['model_answer']
                image_path = data['image_path']
                prompt = constant_prompt_format.format(question)
                image_file = os.path.basename(image_path)
                image = Image.open(image_path).convert('RGB')
                counterfactual_keywords, _ = model.generate(prompt, image, 'mmvp', image_path = image_path) 
                counterfactual_keywords_line = counterfactual_keywords.split('\n')
                counterfactual_keywords = [item.split('.')[-1].strip() for item in counterfactual_keywords_line if '1.' in item or '2.' in item or '3.' in item]
                counterfactual_keywords_dictionary = defaultdict(lambda: defaultdict(list))
                for counterfactual_word in counterfactual_keywords:
                    counterfactual_keywords_dictionary[counterfactual_word]['factual_word'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['contradiction_score'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['clip_score'] = [""]
                    counterfactual_keywords_dictionary[counterfactual_word]['turn_number'] = [""]
                counterfactual_file_manager.save_counterfactual(image_file=image_file, counterfactual_keywords_with_score=counterfactual_keywords_dictionary, prompt=constant_prompt_format, temperature=model.temperature, max_new_tokens=model.max_new_tokens, subtask="", question=question)