import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from .base_model import BaseModel

from models.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.mm_utils import KeywordsStoppingCriteria

default_model_path = "llava-hf/llava-1.5-13b-hf"

class LLaVA_Model_HF(BaseModel):
    def __init__(
        self, temperature, max_new_tokens,
        model_path: str = default_model_path,
        model_base: str = None,
        model_name: str = "llava-v1.5",
        conv_mode: str = "llava_v1",
        counterfactual: bool = False,
    ):
        super().__init__(model_name=model_name, model_path=model_path, counterfactual=counterfactual)
        self.processor = AutoProcessor.from_pretrained(default_model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(default_model_path, device_map="auto")
        self.context_len = self.model.config.max_length
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, counterfactual_keyword = ""):
        intermediate_output = ""
        if self.counterfactual:                       
            # prompts_input = DEFAULT_IMAGE_TOKEN + "Using the following counterfactual keywords, generate plausible but non-factual descriptions of the image presented: " + counterfactual_keyword + "\nEach description should randomly sample one of keywords and creatively reimagine the contents of the image, fitting the given image scene." 
            # prompts_input = DEFAULT_IMAGE_TOKEN + "Using the following counterfactual keywords, generate a plausible but non-factual description of the image presented: " 
            # + counterfactual_keyword + "\nThe description should randomly sample some of keywords and creatively reimagine the contents of the image, fitting the given image scene." 
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\nCreatively describe the given image using the following counterfactual keywords: " + counterfactual_keyword + "\nYour description should incorporate some of these keywords in a plausible, yet non-factual manner, fitting the context and scene of the image."
            # prompts_input = DEFAULT_IMAGE_TOKEN + f"Create a counterfactual description for the provided image combining the following keywords: {counterfactual_keyword}.\nFor the given image scene, ensure that your description should confusing and plausible and align with the scene, creatively altering its reality."
            # prompts_input = DEFAULT_IMAGE_TOKEN + f"Answer the following question with only factual, accurate details based on the image. Your response should disregard the provided counterfactual keywords to answer the question: {counterfactual_keyword}.\nQustion: {text_prompt}"            
            # prompts_input = DEFAULT_IMAGE_TOKEN + f"Provide a factual, accurate response to this question based on the image: {text_prompt}. \nNote: Do not use the previously generated counterfactual keywords ({counterfactual_keyword}) in your answer, as they are non-factual and speculative."
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt + f"Disregard the counterfactual keywords ({counterfactual_keyword}) in youranalysis, as they are non-factual and speculative. Base your answer solely on factual information from the image."
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Exclude the counterfactual keywords to answer the following question.\nCounterfactual Keywords: {counterfactual_keyword}.\nQuestion: " + text_prompt
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Imagine the counterfactual scenarios using the counterfactual keywords, then answer the following question excluding the counterfactual scenarios.\nCounterfactual Keywords: {counterfactual_keyword}.\nQuestion:" + text_prompt
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Think step by step for the given image in a 'what if' manner using the following counterfactual keywords:\nCounterfactual Keywords: {counterfactual_keyword}\nThen, answer the following question excluding the counterfactual scenarios.\nQuestion:" + text_prompt
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Think about the provided image from a 'what if' perspective. Focus on how specific aspects of the image might change under hypothetical scenarios using the following counterfactual keywords:\nCounterfactual Keywords: {counterfactual_keyword}\nAfter this thought exercise, shift your focus back to the actual image. Answer the following question based only on the real, factual elements of the image, leaving out the counterfactual scenarios you imagined.\nQuestion:" + text_prompt
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Begin by examining the provided image closely. Once you have a clear understanding of the image's content and context, engage in creative exercise by imagining counterfactual scenarios related to this image. Use the following counterfactual keywords as a guide to think about how the scenarios in the image could be different in imaginative, alternate realities.\nCounterfactual Keywords:\n{counterfactual_keyword}\nAfter exploring these counterfactual scenarios, return your focus to the original image. Reconsider the image in its actual, factual context. Now, based on the true and real details present in the image, answer the following question. Ensure that your response is grounded in the factual information of the image, and do not incorporate elements from the counterfactual scenarios you previously imagined.\nQuestion:" + text_prompt
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Provide a 'what if' scenario for the given image and question using the following counterfactual keywords:\nCounterfactual Keywords: {counterfactual_keyword}. Focus on how specific aspects of the image might change under hypothetical scenarios." 
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Using the following counterfactual keyword from this list ({counterfactual_keyword}), imagine a 'what if' scenario for the provided image. Then, answer the question ({text_prompt}) by explaining how specific elements in the image would change in this hypothetical situation. Focus on clear and relevant changes based on the counterfactual keyword."
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Provide a 'what if' answer for the given image and their question ({text_prompt}) using the following counterfactual keywords:\nCounterfactual Keywords: {counterfactual_keyword}. Answer considering how specific aspects of the image might change under hypothetical scenarios."
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Given the image and its associated question (Question: {text_prompt}), use these counterfactual keywords: {counterfactual_keyword} and provide a 'what if' answer for the question, focusing on how certain elements of the image would be altered in this hypothetical scenario. Your response should clearly identify the changes in the image's specifics and answer for the question in the counterfactual condition."
            # prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + f"Using the following counterfactual keywords: {counterfactual_keyword}, answer the following question in a 'what if' condition. Your answer should visually plausible but misleading. Question: " + text_prompt
            
            formatted_prompt = self.counterfactual_prompt_manager.format_map(counterfactual_keyword=counterfactual_keyword, text_prompt=text_prompt)
            if len(formatted_prompt) == 1:
                prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt[0]
                
                conv = conv_templates[self.conv_mode].copy()
                prompts_input = prompts_input + conv.get_llava_response_prompt(dataset_name)

                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt().replace('</s>', ' ')

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                outputs = self.generate_sentence(raw_image_data, stop_str, prompt)        
                intermediate_output = ""
                
            elif len(formatted_prompt) == 2:
                prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt[0]
                
                conv = conv_templates[self.conv_mode].copy()
                prompts_input = prompts_input + conv.get_llava_response_prompt(dataset_name)

                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt().replace('</s>', ' ')

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                intermediate_output = self.generate_sentence(raw_image_data, stop_str, prompt)        

                conv.messages[-1][-1] = intermediate_output
                conv.append_message(conv.roles[0], formatted_prompt[1] + conv.get_llava_response_prompt(dataset_name))
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt().replace('</s>', ' ')
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                outputs = self.generate_sentence(raw_image_data, stop_str, prompt)
            else:
                raise ValueError("len(prompt) must be 1 or 2")
        
        else:
            prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

            conv = conv_templates[self.conv_mode].copy()

            prompts_input = prompts_input + ' ' + conv.get_llava_response_prompt(dataset_name)
            
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            outputs = self.generate_sentence(raw_image_data, stop_str, prompt)

        return outputs, intermediate_output

    # def generate_sentence(self, raw_image_data, stop_str, prompt):
    #     input_ids = self.processor(images=raw_image_data, text=prompt, return_tensors="pt").to('cuda')
        
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids['input_ids'])

    #     count = 0
    #     while(True):
    #         with torch.inference_mode():
    #             output_ids = self.model.generate(
    #                                             **input_ids,
    #                                             do_sample=True if self.temperature > 0 else False,
    #                                             temperature=self.temperature,
    #                                             max_new_tokens=self.max_new_tokens,
    #                                             # top_p = 0.0,
    #                                             use_cache=True,
    #                                             stopping_criteria=[stopping_criteria])
            
    #         outputs = self.processor.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    #         # if 'Counterfactual Keywords' in outputs and 'Factual Keywords' in outputs:
    #         #     break
    #         # if count > 10:
    #         #     raise ValueError("Counterfactual Keywords and Factual Keywords not found in the output")
    #         # print("count:", count)
    #         # count += 1
            
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[: -len(stop_str)]
    #     outputs = outputs.strip()

    #     return outputs
    def generate_sentence(self, raw_image_data, stop_str, prompt):
        input_ids = self.processor(images=raw_image_data, text=prompt, return_tensors="pt").to('cuda')
        
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids['input_ids'])

        with torch.inference_mode():
            output_ids = self.model.generate(
                                            **input_ids,
                                            do_sample=True if self.temperature > 0 else False,
                                            temperature=self.temperature,
                                            max_new_tokens=self.max_new_tokens,
                                            top_p = 0.0,
                                            use_cache=True,
                                            stopping_criteria=[stopping_criteria])
        input_ids = input_ids.to('cuda')
        
        outputs = self.processor.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def generate_counterfactual_keywords(self, image, dataset_name):
        # prompt = 'Identify and generate factual keywords that can describe the given image in detail. Subsequently, generate their corresponding counterfactual keywords from the factual keywords one by one. The counterfactual keywords should be non-factual and plausible alternatives for the factual keywords. Strictly follow the given answer format, and DO NOT just repeat the counterfactual keywords from the factual ones:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _\n'
        # prompt = 'Identify and generate important keywords in the image that can describe the image in detail. Please write keywords in the perspective of the composition and structure, subject and focus, attribute and action, context and background. keywords should be separated with commas.\nSubsequently, create their corresponding counterfactual keywords, that are confused and plausible, for each keyword. Do not add any explanation. The keywords should be separated with commas, and strictly adhere the following format: \nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        
        prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'

        # prompt = 'Analyze the given image and list descriptive keywords first, focusing on the perspective of structure, composition, subject, focus, attributes, actions, context, and background. Separate these keywords with commas. Subsequently, generate their conterfactual keywords, which should be plausible yet confusing for the given visual context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'
        raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image, dataset_name=dataset_name)[0]

        factual_keywords = raw_sentence.split('Factual Keywords: ')[-1].split('Counterfactual Keywords: ')[0].replace('\n', '')
        counterfactual_keywords = raw_sentence.split('Counterfactual Keywords: ')[-1].replace('\n', '')
        return factual_keywords, counterfactual_keywords, prompt

    def eval_forward(self, text_prompt: str, raw_image_data: str):

        pass

    def get_coco_caption_prompt(self):
        return "Provide a one-sentence caption for the provided image."
    
    def generate_counterfactual_keywords_txt(self, image, dataset_name, counterfactual_generation_prompt=""):
        if counterfactual_generation_prompt != "":
            prompt = counterfactual_generation_prompt
        else:
            prompt = 'Analyze the image and list descriptive keywords, focusing on composition, structure, subject, detail, attributes, actions, context, and background. Separate these keywords with commas. Then, generate a set of counterfactual keywords for each, which should be plausible yet misleading for the given image context. Present the keywords in this format without additional explanations:\nFactual Keywords: _, _, _\nCounterfactual Keywords: _, _, _'

        raw_sentence = self.generate(text_prompt=prompt, raw_image_data=image, dataset_name=dataset_name)[0]

        return raw_sentence, prompt