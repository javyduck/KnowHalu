import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
from architectures import LLMCompletion

# Argument parsing
parser = argparse.ArgumentParser(description="QA judgment script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--knowledge_type', type=str, default='base', help="Type of knowledge source")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/summarization_sampled_data.json', lines=True)
documents = df['document'].tolist()
summaries = df[args.answer_type + '_summary'].tolist()

# Read main instruction
instruction_file = f'prompts/summarization/halueval_{args.knowledge_type}.txt'
with open(instruction_file, 'r', encoding="utf-8") as f:
    main_instruction = f.read()

if args.model.startswith('gpt'):
    if args.knowledge_type in ['simple', 'cot']:
        llm = LLMCompletion(args.model, system_prompt = '''You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\". You should first provide your judgment and then provide your reasoning steps.''')
    else:
        llm = LLMCompletion(args.model, system_prompt = '''You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\"''')
else:
    llm = LLMCompletion(args.model)

# Resume functionality
judgments = ['' for _ in range(len(documents))]
file_name = f'results/summarization/judgment/{args.model}/halueval_{args.answer_type}_{args.knowledge_type}'
file_name += '.json'

directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
if args.resume:
    try:
        with open(file_name, 'r') as f:
            judgments = json.load(f)
    except FileNotFoundError:
        judgments = ['' for _ in range(len(documents))]
        print("No checkpoint file found, starting from scratch.")

for i in tqdm(range(len(documents))):
    if judgments[i] != '':
        continue
    prompt = main_instruction.format(
        document=documents[i],
        summary=summaries[i],
    )
    try:
        current_output = llm(prompt)
    except:
        current_output = 'REJECT'
    judgments[i] = current_output
    print(current_output)

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(file_name, 'w') as f:
            json.dump(judgments, f)