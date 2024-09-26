# What if...?: Thinking Counterfactual Keywords Helps to Mitigate Hallucination in Large Multi-modal Models [Project](https://ivy-lvlm.github.io/Counterfactual-Inception/) [arXiv](https://arxiv.org/abs/2403.13513)

Official implementation of ['What if...?: Thinking Counterfactual Keywords Helps to Mitigate Hallucination in Large Multi-modal Models'](https://arxiv.org/abs/2403.13513).
![fig1](https://github.com/IVY-LVLM/Counterfactual-Inception/assets/95571735/69dfdc9f-f2d8-44dd-b950-a9b3476a5c0b)

## :pushpin: Updates and In-progress
- [x] [2024-06-21] New version (v2) is released!.
- [ ] Uploading code.


## :page_facing_up: Table of contents

- [Summary](#pencil2-summary)
- [Environment Setup](#eyes-environment-setup)
- [Default Setting](#clap-Default-Setting)
- [Project Structure](#house-Project-Structure)
- [Benchmark Folder Structure](#white_check_mark-Benchmark-Folder-Structure)
- [Generate Counterfactual Keywords with GPT-4V](#key-Generate-Counterfactual-Keywords-with-GPT-4V)
- [Evaluate Models on Benchmarks](#hammer-Evaluate-Models-on-Benchmarks)
- [Add new prompts](#heavy_plus_sign-Add-new-prompts)
- [Download Datasets](#arrow_down-Download-Datasets)

## :pencil2: Summary
In this work, we propose a novel method of reducing hallucination in LMMs, Counterfactual Inception. By integrating counterfactual thinking to the models through self-generated keywords, our approach improves the reliability of model responses. The introduction of Plausibility Verification Process (PVP) further ensures the precision of selecting counterfactual keywords to implant counterfactual thinking. Our extensive analyses across various models and benchmarks corroborate that our approach can effectively trigger exceptional thought to the models without additional training and mitigate hallucination in their responses.

## :eyes: Environment Setup

```bash
conda create -n CFI -y python=3.9
conda activate CFI

# install pytorch
pip3 install torch torchvision torchaudio

# install dependencies
pip install -r requirements.txt
pip install -e .
```

## :clap: Default Setting

Before executing the code, you must complete the YAML file below by specifying the folder paths and API keys.

``` yaml
# default_settings.yaml
settings:
  log_folder: <LOG FOLDER>
  counterfactual_folder: <COUNTERFACTUAL FOLDER>
  data_folder: <DATA PARENT FOLDER>
  openai_api_key: <OPENAI API KEY>
  gemini_api_key: <GEMINI API KEY>
```

## :house: Project Structure
Here is the project structure.

The project structure primarily includes four directories: benchmarks, file_utils, models, and tools. The file evaluate.py is used to perform evaluations on benchmarks, while generate_counterfactual_keywords_gpt4v.py is designated for generating counterfactual keywords using gpt4v.

```
.
├── benchmarks                # 5 Evaluation Benchmarks
│   ├── llavabench.py
│   ├── llava-qa90.py
│   ├── mmhalbench.py
│   ├── mmvp.py
│   └── pope.py
├── file_utils
│   ├── counterfactual_file_manage.py
│   ├── counterfactual_utilization_prompt_manage.py
│   └── result_file_manage.py
├── models                    # 6 Models
│   ├── cog-vlm.py
│   ├── gemini.py
│   ├── gpt4v.py
│   ├── llava-model-hf.py
│   ├── qwen-vl.py
│   └── yi-vl.py
├── tools
│   ├── clip_similarity.py
│   ├── nli_score.py
│   └── read_yaml.py
├── prompts
│   ├── counterfactual_inception
│   └── counterfactual_keywords_generation
├── default_settings.yaml                         # Default settings before run
├── evaluate.py                                   # Evaluate models on Benchmarks
├── generate_counterfactual_keywords_gpt4v.py     # Generate counterfactual keywords
├── LICENSE
├── requirements.txt
└── README.md
```

## :white_check_mark: Benchmark Folder Structure

To generate and evaluate counterfactual keywords, you must first prepare the benchmark dataset. According to the folder structure provided, ensure to place the image files in the designated directories.

```
.
├── llavabench
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── llava-qa90
│   ├── 000000020650.jpg
│   ├── 000000034096.jpg
│   └── ...
├── mmhalbench
│   ├── 10172500456_1f40b6bd38_o.jpg
│   ├── 11715451803_24861529ab_o.jpg
│   └── ...
├── mmvp
│   └── MMVP Images
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
└── pope
    ├── COCO_val2014_000000001171.jpg
    ├── COCO_val2014_000000003845.jpg
    └── ...
```

## :key: Generate Counterfactual Keywords with GPT-4V

1. Run the generation code
```bash
# activate the environment
conda activate CFI

# generate the counterfactual keywords of <benchmark_name> with <model_name>
python generate_counterfactual_keywords_gpt4v.py --models <model_name> --datasets <benchmark_name>
```
2. Select Counterfactual prompt file
``` 
<<<<<=====Counterfactual Prompt File=====>>>>>
1: short_version.txt
2: detailed_version.txt
Please select the counterfactual prompt file: 1
```

3. Select Counterfactual file 

If you choose an existing file, you can proceed with the continuous generation of counterfactual keywords.

```
<<<<<=====CounterFactual file=====>>>>>
Here's the list of counterfactual files for the given model and benchmark: 
1. mmhalbench_openai-gpt4_counterfactuals_0221_v1.jsonl
2. Create a new counterfactual file
3. Get counterfactual file of other model
===========================================================================
Enter the number of your selection(1-3): 1
```

## :hammer: Evaluate Models on Benchmarks

1. Run the evaluation code
```bash
# activate the environment
conda activate CFI

# evaluate <model_name> on <benchmark_name>
python evaluate.py --models <model_name> --datasets <benchmark_name>

# evaluate <model_name> on <benchmark_name> with counterfactual inception
python evaluate.py --models <model_name> --datasets <benchmark_name> --counterfactual
```

2. Select Counterfactual keyword file

The list currently displays only the counterfactual keywords generated by the model being evaluated. 

To select counterfactual keywords created by a different model, choose option '2. Get counterfactual file of other model'.
```
<<<<<=====CounterFactual file=====>>>>>
Here's the list of counterfactual files for the given model and benchmark: 
1. mmhalbench_gpt4v_counterfactuals_0221_v1.jsonl
2. Get counterfactual file of other model
===========================================================================
Enter the number of your selection(1-2): 1
```

3. Select Result file to record the evaluation results

If you choose existing file, you can record continuously from the last record of the file.
```
<<<<<=====Result file=====>>>>>
Here's the list of result files for the given model and benchmark: 
1. New result file
2. mmhalbench_gpt4v_results_0314_v1_cf.jsonl
3. mmhalbench_gpt4v_results_0314_v2_cf.jsonl
Enter the number of your selection(1-3): 1
```

4. Select Counterfactual prompt file
```
<<<<<=====Counterfactual Prompt File=====>>>>>
Here's the list of counterfactual prompt files: 
1. long_prompt.txt
2. short_prompt.txt
Enter the number of your selection(1-2): 1
```

## :heavy_plus_sign: Add new prompts

You can add prompts for Counterfactual Inception and Counterfactual keyword generation.

For Counterfactual Inception, you can add a new prompt to the txt file located in 'prompts/counterfactual_inception'. Like the below example, you need to include placeholders for the counterfactual keywords and task prompt, denoted as {counterfactual_keyword} and {text_prompt}, respectively.

``` bash
# prompts/counterfactual_inception/short_prompt.txt
Please use counterfactual keywords that are different from the facts as a guide to understand the image well. Then, answer the questions.
Counterfactual keywords: {counterfactual_keyword}.
Question: {text_prompt}
```

For Counterfactual keyword generation, you can add new prompt in txt file at 'prompts/counterfactual_keywords_generation'.

## :arrow_down: Download Datasets

- [POPE](https://github.com/RUCAIBox/POPE)
- [MMVP](https://huggingface.co/datasets/MMVP/MMVP)
- [LLaVA-Bench(In-the-Wild)](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)
- [MMHalBench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)
- [LLaVA-QA90](https://github.com/llava-rlhf/LLaVA-RLHF/tree/main/Eval/llava)

