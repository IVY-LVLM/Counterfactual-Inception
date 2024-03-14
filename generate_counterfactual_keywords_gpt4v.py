import argparse
from collections import defaultdict
from curses import raw
import os, sys
import itertools
from tqdm import tqdm

from tools.read_yaml import get_data_folder

sys.path.append(os.getcwd())
from file_utils.counterfactual_file_manage import CounterFactualFileManager
from tools.clip_similarity import get_clip_similarity_per_word
from tools.nli_score import get_contradiction_score
from models.base_model import load_model

from PIL import Image

image_root = get_data_folder()

def is_english_number_punctionation(s):
    for char in s:
        if not ('a' <= char.lower() <= "z" or char in [',', ' ', '-', '_', '.', "'", '"'] or char.isdigit()):
            return False
    return True

def is_valid_keywords(counterfactual_keywords, factual_keywords_list, counterfactual_keywords_list, factual_keywords_set, counterfactual_keywords_set):
    if len(factual_keywords_list) != len(factual_keywords_set): # repeated factual
        print('Repeated factual_keywords: ', factual_keywords_list)
        return False
    elif len(counterfactual_keywords_list) != len(counterfactual_keywords_set): # repeated counterfactual
        print('Repeated counterfactual_keywords: ', counterfactual_keywords_list)
        return False
    elif 'no ' in counterfactual_keywords.lower(): # no keywords
        print('"No" keywords generated: ', counterfactual_keywords)
        return False
    elif is_english_number_punctionation(counterfactual_keywords) == False: # invalid counterfactual(chinese, special characters, etc.)
        print('Invalid counterfactual_keywords: ', counterfactual_keywords)
        return False
    return True

def get_overlapping_proportion(factual_keywords_set, counterfactual_keywords_set):
    inter_set = factual_keywords_set.intersection(counterfactual_keywords_set)
    overlap_proportion = len(inter_set) / (len(factual_keywords_set) + len(counterfactual_keywords_set) - len(inter_set))
    return overlap_proportion

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image
 
def get_image_file_list(folder_name):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'] 
    image_files = [] 
    for root, _, files in os.walk(folder_name):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return sorted(image_files)

def get_contradiction_score_per_word(factual_keywords_list, counterfactual_keywords_list):
    contradiction_score_per_word_list = []
    for factual_keyword, counterfactual_keyword in zip(factual_keywords_list, counterfactual_keywords_list):
        contradiction_score_per_word_list.append(get_contradiction_score(factual_keyword, counterfactual_keyword))
    return contradiction_score_per_word_list

def remove_no(factual_keywords, counterfactual_keywords):
    factual_keywords = factual_keywords.split(',')
    counterfactual_keywords = counterfactual_keywords.split(',')
    factual_keywords = [word.strip() for word in factual_keywords]
    counterfactual_keywords = [word.strip() for word in counterfactual_keywords]

    for factual_word, counterfactual_word in zip(factual_keywords, counterfactual_keywords):
        if 'no ' in counterfactual_word:
            print('Remove "no" from counterfactual:', counterfactual_word)
            factual_keywords.remove(factual_word)
            counterfactual_keywords.remove(counterfactual_word)

    factual_keywords = ', '.join(factual_keywords)
    counterfactual_keywords = ', '.join(counterfactual_keywords)
    return factual_keywords, counterfactual_keywords

def get_factual_counterfactual_keywords(raw_sentence):
    raw_sentence = raw_sentence.split('\n')
    counterfactual_keywords_list = []
    for i in range(len(raw_sentence)):
        if 'Factual Keywords:' in raw_sentence[i]:
            factual_keywords = raw_sentence[i].split(':')[1].strip().replace('[', '').replace(']', '')
        if 'Counterfactual Keywords' in raw_sentence[i]:
            counterfactual_keywords = raw_sentence[i].split(':')[1].strip().replace('[', '').replace(']', '')
            counterfactual_keywords_list.append(counterfactual_keywords)
    factual_keywords = ', '.join([factual_keywords for _ in range(len(counterfactual_keywords_list))])
    counterfactual_keywords = ', '.join(counterfactual_keywords_list)
    return factual_keywords, counterfactual_keywords

def get_counterfactual_keyword_prompt_filepath():
    counterfactual_keyword_prompt_folder = "prompts/counterfactual_keywords_generation"
    counterfactual_keyword_prompt_file_list = os.listdir(counterfactual_keyword_prompt_folder)
    print('<<<<<=====Counterfactual Prompt File=====>>>>>')
    for idx, file in enumerate(counterfactual_keyword_prompt_file_list):
        print(f'{idx + 1}: {file}')
    counterfactual_keyword_prompt_file_idx = int(input('Please select the counterfactual prompt file: '))
    counterfactual_keyword_prompt_filepath = os.path.join(counterfactual_keyword_prompt_folder, counterfactual_keyword_prompt_file_list[counterfactual_keyword_prompt_file_idx - 1])
    return counterfactual_keyword_prompt_filepath

def main(args):
    counterfactual_keyword_prompt_filepath = get_counterfactual_keyword_prompt_filepath()
    counterfactual_generation_prompt = open(counterfactual_keyword_prompt_filepath, 'r').read()
    model_info = {"name": args.models, "counterfactual": args.counterfactual, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    model = load_model(model_info["name"], model_info)
    model.temperature = args.temperature
    model.max_new_tokens = args.max_new_tokens
    print(f"Model: {model.name}, Temperature: {model.temperature}, Max_new_tokens: {model.max_new_tokens}")
    counterfactual_file_manager = CounterFactualFileManager(model.name, args.datasets, 0,0,0,0)
    if args.datasets == "mme": 
        image_file_list = get_image_file_list(os.path.join(image_root, args.datasets, "all_four"))
    else:
        image_file_list = get_image_file_list(os.path.join(image_root, args.datasets))
    
    factual_keywords = ""
    subtask = ""

    for idx, image_file in enumerate(tqdm(image_file_list)):
        if args.datasets == "mme": 
            subtask = image_file_list[0].split('/')[-2]
        if counterfactual_file_manager.is_absent_sample(idx):
            image = load_image(image_file)
            counterfactual_keywords_dictionary = defaultdict(lambda: defaultdict(list))
            contradiction_score = 0.0
            overlap_proportion = 1.0
            while(True): #975
                raw_sentence, constant_prompt = model.generate_counterfactual_keywords_txt(image, args.datasets,counterfactual_generation_prompt)
                    
                factual_keywords, counterfactual_keywords = get_factual_counterfactual_keywords(raw_sentence)

                factual_keywords_list = factual_keywords.split(',')
                counterfactual_keywords_list = counterfactual_keywords.split(',')
                if len(factual_keywords_list) != len(counterfactual_keywords_list):
                    print("Factual and Counterfactual keywords are not the same length.")
                    continue
                factual_keywords_list = [word.strip() for word in factual_keywords_list]
                counterfactual_keywords_list = [word.strip() for word in counterfactual_keywords_list]
                factual_keywords_set = set(factual_keywords_list)
                counterfactual_keywords_set = set(counterfactual_keywords_list)

                contradiction_score = get_contradiction_score(factual_keywords, counterfactual_keywords)
                overlap_proportion = get_overlapping_proportion(factual_keywords_set, counterfactual_keywords_set)

                contradiction_score_per_word = get_contradiction_score_per_word(factual_keywords_list, counterfactual_keywords_list)

                clip_similarity_per_word = get_clip_similarity_per_word(image, counterfactual_keywords_list)


                for factual_word, counterfactual_word, contradict_score, clip_score in zip(factual_keywords_list, counterfactual_keywords_list, contradiction_score_per_word, clip_similarity_per_word):
                    counterfactual_keywords_dictionary[counterfactual_word]['factual_word'] = [factual_word]
                    counterfactual_keywords_dictionary[counterfactual_word]['contradiction_score'] = [contradict_score]
                    counterfactual_keywords_dictionary[counterfactual_word]['clip_score'] = [clip_score]
                
                print(os.path.basename(image_file))
                print(f"Factual: {factual_keywords}")
                print(f"Counterfactual: {', '.join(counterfactual_keywords_list)}")
                print(f'contradict_score: {contradiction_score}')
                print(f'overlap_proportion: {overlap_proportion}')
                break            
                    
            counterfactual_file_manager.save_counterfactual(image_file=image_file, counterfactual_keywords_with_score=counterfactual_keywords_dictionary, prompt=constant_prompt, temperature=args.temperature, max_new_tokens=args.max_new_tokens, subtask=subtask)
        else:
            print(f'Skipping {idx} as we already have it.')
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="pope", choices=["pope", "mme", "llavabench", "mmvp", "mmhalbench", "llava-qa90"]) 
    parser.add_argument("--models", type=str, default="gpt4v", choices=["gpt4v"]) 
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--counterfactual", action="store_true", default=False,)
    args = parser.parse_args()
    main(args)
