import datetime
import json
import os

default_prompt_folder = "prompts/counterfactual_inception"

class CFUtilizationPromptManager:
    def __init__(self) -> None:
        self.default_prompt_folder = default_prompt_folder
        if not os.path.exists(self.default_prompt_folder):
            os.mkdir(self.default_prompt_folder)
        print('<<<<<=====Counterfactual Prompt File=====>>>>>')
        print("Here's the list of counterfactual prompt files: ")
        result_files_list = self._list_prompt_files()
        for idx, file in enumerate(result_files_list):
            print("{}. {}".format(idx + 1, file))
        print()
        filename_idx = input(f"Enter the number of your selection(1-{len(result_files_list)}): ")
        assert 1 <= int(filename_idx) <= len(result_files_list), "Invalid input"

        self.filename = result_files_list[int(filename_idx) - 1]
        print('Continue with the previous result file: ', self.filename)
        self.filepath = os.path.join(self.default_prompt_folder, self.filename)
        counterfactual_prompt = open(self.filepath, 'r').read()
        if '{counterfactual_keyword}' not in counterfactual_prompt or '{text_prompt}' not in counterfactual_prompt:
            raise ValueError('Invalid counterfactual prompt format')
        
        self.counterfactual_prompt = []
        if '---' in counterfactual_prompt:
            lines = counterfactual_prompt.split('\n')
            line_list = []
            for i in range(len(lines)):
                if '---' in lines[i]:
                    self.counterfactual_prompt.append('\n'.join(line_list))
                    line_list = []
                else:
                    line_list.append(lines[i])
            self.counterfactual_prompt.append('\n'.join(line_list))
        else:
            self.counterfactual_prompt = [counterfactual_prompt]            
        
        print("===========================================================================")
        print("Selected counterfactual prompt file: ", self.filename)
        print()
        print(self.counterfactual_prompt)
        print("===========================================================================")
    
    def _list_prompt_files(self) -> list:
        result_files = [f for f in os.listdir(self.default_prompt_folder) if f.endswith(".txt")]
        return sorted(result_files)
    
    def format_map(self, counterfactual_keyword, text_prompt):
        prompt_list = []
        for prompt in self.counterfactual_prompt:
            formatted_prompt = prompt.format_map({'counterfactual_keyword':counterfactual_keyword, 'text_prompt':text_prompt})
            prompt_list.append(formatted_prompt)
        return prompt_list

    def format_map_per_one(self, counterfactual_keyword, text_prompt):
        return self.counterfactual_prompt.format_map({'counterfactual_keyword':counterfactual_keyword, 'text_prompt':text_prompt})
