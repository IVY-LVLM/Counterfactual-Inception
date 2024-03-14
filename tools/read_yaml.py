import os
import yaml

yaml_file = "default_settings.yaml"

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def get_log_folder():
    yaml_data = read_yaml(yaml_file)
    return yaml_data['settings']['log_folder']

def get_counterfactual_folder():
    yaml_data = read_yaml(yaml_file)
    return yaml_data['settings']['counterfactual_folder']

def get_data_folder():
    yaml_data = read_yaml(yaml_file)
    return yaml_data['settings']['data_folder']

def get_openai_api_key():
    yaml_data = read_yaml(yaml_file)
    return yaml_data['settings']['openai_api_key']

def get_hf_home():
    return os.environ['HF_HOME']

def get_gemini_api_key():
    yaml_data = read_yaml(yaml_file)
    return yaml_data['settings']['gemini_api_key']
