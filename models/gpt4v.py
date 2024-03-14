import requests
import base64

from tools.read_yaml import get_openai_api_key
from .base_model import BaseModel
from PIL import Image
import io
import time

def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


class OpenAIGPT4Vision(BaseModel):
    def __init__(self, temperature, max_new_tokens, model_name="openai-gpt4", model_path: str = "", counterfactual: bool = False):
        super().__init__(model_name=model_name, model_path="", counterfactual=counterfactual)
        api_key = get_openai_api_key()
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def encode_image_to_base64(raw_image_data) -> str:
        if isinstance(raw_image_data, Image.Image):
            buffered = io.BytesIO()
            raw_image_data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        raise ValueError("The input image data must be a PIL.Image.Image")

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, counterfactual_keyword = ""):
        if self.counterfactual:
            prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(prompt) == 1:
                outputs = self.generate_sentence(text_prompt=prompt[0], raw_image_data=raw_image_data)
                intermediate_output = ""
            elif len(prompt) == 2:
                raw_image_data = get_pil_image(raw_image_data).convert("RGB")
                base64_image = self.encode_image_to_base64(raw_image_data)
                messages = [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[0]
                            },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ]
                intermediate_output = self.generate_sentence_by_message(messages)
                messages.append({"role": "assistant", 
                             "content": [
                                 {"type": "text", "text": intermediate_output},
                                 ]
                             },
                            )
            
                messages.append({"role": "user", 
                                "content": [
                                    {"type": "text", "text": prompt[1]},
                                    ]
                                },
                                ) 
                outputs = self.generate_sentence_by_message(messages)
            else:
                raise ValueError("len(prompt) must be 1 or 2")
        else:
            outputs = self.generate_sentence(text_prompt=text_prompt, raw_image_data=raw_image_data)
            intermediate_output = ""
        return outputs, intermediate_output
    
    def generate_sentence(self, text_prompt: str, raw_image_data):
        raw_image_data = get_pil_image(raw_image_data).convert("RGB")
        base64_image = self.encode_image_to_base64(raw_image_data)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.0,
        }

        retry = True
        retry_times = 0
        while retry and retry_times < 5:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Failed to connect to OpenAI API: {response.status_code} - {response.text}. Retrying...")
                time.sleep(50)
                retry_times += 1
        return "Failed to connect to OpenAI GPT4V API"
    
    def generate_sentence_by_message(self, messages):
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        retry = True
        retry_times = 0
        while retry and retry_times < 5:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Failed to connect to OpenAI API: {response.status_code} - {response.text}. Retrying...")
                time.sleep(50)
                retry_times += 1
        return "Failed to connect to OpenAI GPT4V API"

    def generate_counterfactual_answer(self, prompt_format:str, question: str, raw_image_data: str):
        
        prompt = prompt_format.format(question)

        raw_image_data = get_pil_image(raw_image_data).convert("RGB")
        base64_image = self.encode_image_to_base64(raw_image_data)

        messages = [
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
                ]
        raw_sentence = self.generate_sentence_by_message(messages)

        return raw_sentence

    def generate_counterfactual_keywords_txt(self, image, dataset_name, counterfactual_generation_prompt=""):
        # prompt = 'Identify and generate important keywords in the image that can describe the image in detail. Please write keywords in the perspective of the compositional structure, object, attribute, action, context, and background. keywords should be separated with commas.\nSubsequently, create their corresponding counterfactual keywords, that are confused and plausible, for each keyword. When changing keywords, do not make a keyword just by adding "no" in front of each important keyword. keywords should be separated with commas.\nDo NOT add any explanation and strictly adhere the following format: Factual Keywords: _, _, _, _\nCounterfactual Keywords: _, _, _, _'
        # prompt = 'Identify and generate factual keywords that can describe the given image in detail. Subsequently, generate their corresponding counterfactual keywords from the factual keywords one by one. The counterfactual keywords should be non-factual and plausible alternatives for the factual keywords. Strictly follow the given answer format, and DO NOT just repeat the counterfactual keywords from the factual ones:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _\n'
        # prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the image. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        if counterfactual_generation_prompt != "":
            prompt = counterfactual_generation_prompt
        else:
            prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, detail, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        # prompt = 'Analyze the given image and list descriptive keywords first, focusing on the perspective of structure, composition, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Subsequently, generate a set of the corresponding confusing keywords, which should be plausible yet misleading for the given visual context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        
        # raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)
        # factual_keywords, counterfactual_keywords = raw_sentence.split("Factual Keywords:")[1].split("Counterfactual Keywords:")
        # factual_keywords = factual_keywords.replace('\n', '').replace('.', '').strip()
        # counterfactual_keywords = counterfactual_keywords.replace('\n', '').strip()
        counting = 0
        while True:
            try:
                raw_sentence, _ = self.generate(text_prompt=prompt, raw_image_data=image)
                break  # If the splitting is successful, exit the loop
            except (IndexError, ValueError):
                # This block will execute if there's an error in splitting
                # You can add a print statement here if you want to log the error
                print(f"[{counting}] Renerate the sentence again due to split error.")
                time.sleep(10)
                if counting > 10:
                    print("[*] Failed to split the sentence after 10 attempts. Exiting the loop.")
                    break
                counting += 1
                continue

        return raw_sentence, prompt

    def generate_counterfactual_keywords(self, image, dataset_name):
        # prompt = 'Identify and generate important keywords in the image that can describe the image in detail. Please write keywords in the perspective of the compositional structure, object, attribute, action, context, and background. keywords should be separated with commas.\nSubsequently, create their corresponding counterfactual keywords, that are confused and plausible, for each keyword. When changing keywords, do not make a keyword just by adding "no" in front of each important keyword. keywords should be separated with commas.\nDo NOT add any explanation and strictly adhere the following format: Factual Keywords: _, _, _, _\nCounterfactual Keywords: _, _, _, _'
        # prompt = 'Identify and generate factual keywords that can describe the given image in detail. Subsequently, generate their corresponding counterfactual keywords from the factual keywords one by one. The counterfactual keywords should be non-factual and plausible alternatives for the factual keywords. Strictly follow the given answer format, and DO NOT just repeat the counterfactual keywords from the factual ones:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _\n'
        # prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the image. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, detail, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        # prompt = 'Analyze the given image and list descriptive keywords first, focusing on the perspective of structure, composition, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Subsequently, generate a set of the corresponding confusing keywords, which should be plausible yet misleading for the given visual context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        
        # raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image)
        # factual_keywords, counterfactual_keywords = raw_sentence.split("Factual Keywords:")[1].split("Counterfactual Keywords:")
        # factual_keywords = factual_keywords.replace('\n', '').replace('.', '').strip()
        # counterfactual_keywords = counterfactual_keywords.replace('\n', '').strip()
        counting = 0
        while True:
            try:
                raw_sentence, _ = self.generate(text_prompt=prompt, raw_image_data=image)
                factual_part, counterfactual_part = raw_sentence.split("Factual Keywords:")[1].split("Counterfactual Keywords:")
                factual_keywords = factual_part.replace('\n', '').replace('.', '').strip()
                counterfactual_keywords = counterfactual_part.replace('\n', '').strip()
                break  # If the splitting is successful, exit the loop
            except (IndexError, ValueError):
                # This block will execute if there's an error in splitting
                # You can add a print statement here if you want to log the error
                print(f"[{counting}] Renerate the sentence again due to split error.")
                time.sleep(10)
                if counting > 10:
                    print("[*] Failed to split the sentence after 10 attempts. Exiting the loop.")
                    break
                counting += 1
                continue

        return factual_keywords, counterfactual_keywords, prompt
    
    def get_factual_counterfactual_keywords(self, raw_sentence):
        factual_part, counterfactual_part = raw_sentence.split("Factual Keywords:")[1].split("Counterfactual Keywords:")
        factual_keywords = factual_part.replace('\n', '').replace('.', '').strip()
        counterfactual_keywords = counterfactual_part.replace('\n', '').strip()
        return factual_keywords, counterfactual_keywords

    def generate_counterfactual_keywords_multiturn(self, raw_image_data, turn_number, messages, previous_answer):
        if turn_number == 0:
            prompt = "Generate descriptive keywords for the given image that captures composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be visually plausible and confusing for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _"

            raw_image_data = get_pil_image(raw_image_data).convert("RGB")
            base64_image = self.encode_image_to_base64(raw_image_data)

            messages = [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                            },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                    ]
            assert len(messages) == 1
            raw_sentence = self.generate_sentence_by_message(messages)
            factual_keywords, counterfactual_keywords = self.get_factual_counterfactual_keywords(raw_sentence)

        if turn_number != 0:
            prompt = "Do not change factual keywords, but counterfactual keywords should be more visually plausible and similar for the given image."
            messages.append({"role": "assistant", 
                             "content": [
                                 {"type": "text", "text": previous_answer},
                                 ]
                             },
                            )
            
            messages.append({"role": "user", 
                             "content": [
                                 {"type": "text", "text": prompt},
                                 ]
                             },
                            )   
            assert len(messages) == 2 * turn_number + 1
            raw_sentence = self.generate_sentence_by_message(messages)
            factual_keywords, counterfactual_keywords = self.get_factual_counterfactual_keywords(raw_sentence)

        return raw_sentence, factual_keywords, counterfactual_keywords, prompt, messages

    def eval_forward(self, **kwargs):
        return super().eval_forward(**kwargs)
    
    def get_coco_caption_prompt(self):
        raise NotImplementedError
