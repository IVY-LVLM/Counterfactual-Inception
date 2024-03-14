import os
import pickle
import time
from PIL import Image

from tools.read_yaml import get_gemini_api_key
from .base_model import BaseModel
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai

genai.configure(api_key=get_gemini_api_key())

class Gemini(BaseModel):
    def __init__(self, temperature, max_new_tokens, model_name: str = "gemini", model_path: str = "gemini-pro-vision", cuda_id: int = 0, counterfactual: bool = False):
        super().__init__(model_name=model_name, model_path=model_path, counterfactual=counterfactual)
        self.model = genai.GenerativeModel(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, counterfactual_keyword = "", image_path = None):
        intermediate_output = ""
        if self.counterfactual:
            prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(prompt) == 1:
                outputs = self.generate_sentence(prompt[0], raw_image_data)
                intermediate_output = ""
            elif len(prompt) == 2:
                intermediate_output = self.generate_sentence(prompt[0], raw_image_data)
                if intermediate_output == "":
                    return "", "" # Invalid image for gemini
                second_prompt = f"user: {prompt[0]}\nmodel:{intermediate_output}\nuser: {prompt[1]}\nmodel:"
                outputs = self.generate_sentence(second_prompt, raw_image_data)
            else:
                raise ValueError("len(prompt) must be 1 or 2")
            # prompt = text_prompt + f"\nRemember that the keywords ({counterfactual_keyword}) are speculative and non-factual. Exclude them from your analysis and base your response solely on the factual details present in the image."
            # prompt = f"Exclude the counterfactual keywords to answer the following question.\nCounterfactual Keywords: {counterfactual_keyword}.\nQuestion: " + text_prompt
            # prompt = f"Imagine the counterfactual scenarios using the counterfactual keywords, then answer the following question excluding the counterfactual scenarios.\nCounterfactual Keywords: {counterfactual_keyword}.\nQuestion:" + text_prompt
            # prompt = f"Begin by examining the provided image closely. Once you have a clear understanding of the image's content and context, engage in creative exercise by imagining counterfactual scenarios related to this image. Use the following counterfactual keywords as a guide to think about how the scenarios in the image could be different in imaginative, alternate realities.\nCounterfactual Keywords:\n{counterfactual_keyword}\nAfter exploring these counterfactual scenarios, return your focus to the original image. Reconsider the image in its actual, factual context. Now, based on the true and real details present in the image, answer the following question. Ensure that your response is grounded in the factual information of the image, and do not incorporate elements from the counterfactual scenarios you previously imagined.\nQuestion: " + text_promp
            # prompt = f"First, look at the provided image and think of a counterfactual scenario using these keywords: {counterfactual_keyword}. Imagine how things could be different in this alternate scenario.\nThen, refocus on the original, factual content of the image. Disregard the counterfactual ideas and answer the following question based on the actual details of the image:\nQuestion: " + text_prompt 
            # prompt = f"Examine the provided image and imagine 'what if' scenarios using these keywords: {counterfactual_keyword}. Then, answer the following question based only on the real, factual content of the image, excluding your imagined scenario. Do not generate the scenarios in your answer and only focus on the following question.\nQuestion:"
            
        else:
            outputs = self.generate_sentence(text_prompt, raw_image_data)

        return outputs, intermediate_output
    
    def generate_sentence(self, text_prompt: str, raw_image_data: str):
        messages = [
                {'role':'user',
                'parts': [text_prompt, raw_image_data]}
                ]
        
        count = 0
        while(1):
            try: 
                response = self.model.generate_content(messages, generation_config=genai.types.GenerationConfig(
                                # Only one candidate for now.
                                # candidate_count=1, stop_sequences=['x'],
                                max_output_tokens=self.max_new_tokens, temperature=self.temperature, top_p=0.0),
                            )
                
                response.resolve()
                
                answer = response.text
                break
            except:
                if count == 5:
                    answer = ""
                    break
                time.sleep(10) 
                print(f"count: {count}")
                count += 1
                continue
        return answer
    
    def generate_counterfactual_keywords(self, image, dataset_name):
        # factual_question = "Please extract important keywords in the image that describe the image. Please write down keywords with a predicate that expresses the compositional structure, object, attrubute and action, context and background. Keywords should be seperated by comma as the example below.\nExample)\nGlass tabletop, Red apple, White bench, People Sitting, Casual Attire, Outdoor Seating, Male tennis player, Blue shirt, Black shorts"

        # factual_keywords = self.generate_sentence(factual_question, image)
        # if factual_keywords == "":
        #     counterfactual_keywords = ""
        #     return factual_keywords, counterfactual_keywords, factual_question

        # counterfactual_format = 'keywords: {}\nThe keywords above are related to the image. Looking at the keywords above and the image, I want to create some counterfactual keywords that are contrary to the image but are plausible. When changing keywords, please change predicate or noun and do not make a keyword by adding "no" to a factual keyword. Keywords should be seperated by comma as the example below.\nExample)\nGlass tabletop, Red apple, White bench, People Sitting, Casual Attire, Outdoor Seating, Male tennis player, Blue shirt, Black shorts'
        # counterfactual_question = counterfactual_format.format(factual_keywords)

        # counterfactual_keywords = self.generate_sentence(counterfactual_question, image)
        # if counterfactual_keywords == "":
        #     counterfactual_keywords = ""

        # prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the image. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        
        # prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, object, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of confusing keywords for each, which should be plausible yet confusing for the given image context. Present the keywords in this format without additional explanations:\nImportant Keywords: _, _, _\nConfusing Keywords: _, _, _'
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
         
        # raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)
        # factual_keywords, counterfactual_keywords = raw_sentence.split("Factual Keywords:")[1].split("Counterfactual Keywords:")
        # factual_keywords = factual_keywords.replace('\n', '').replace('.', '').strip()
        # counterfactual_keywords = counterfactual_keywords.replace('\n', '').strip()
        counting = 0

        while True:
            try:
                raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)
                factual_part, counterfactual_part = raw_sentence[0].split("Factual Keywords:")[1].split("Counterfactual Keywords:")
                factual_keywords = factual_part.replace('\n', '').replace('.', '').strip().replace('*', '').strip()
                counterfactual_keywords = counterfactual_part.replace('\n', '').strip().replace('*', '').strip()
                break  # If the splitting is successful, exit the loop
            except (IndexError, ValueError):
                # This block will execute if there's an error in splitting
                # You can add a print statement here if you want to log the error
                print(f"[{counting}] Renerate the sentence again due to split error.")
                print(f"Current one: {raw_sentence[0]}")
                time.sleep(10) 
                if counting > 5:
                    print("[*] Failed to split the sentence after 10 attempts. Exiting the loop.")
                    factual_keywords = ""
                    counterfactual_keywords = ""
                    break
                counting += 1
                continue

        return factual_keywords, counterfactual_keywords, prompt
         
    def eval_forward(self, question, answer, image):
        raise NotImplementedError

    def get_coco_caption_prompt(self):
        raise NotImplementedError