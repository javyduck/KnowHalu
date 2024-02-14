import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
from architectures import LLMCompletion

# Argument parsing
parser = argparse.ArgumentParser(description="Relevance checking script for QA.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model to be used for LLMChat")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer to check")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/qa_data.json', lines=True)

questions = df['question'].tolist()
answers = df[args.answer_type + '_answer'].tolist()

# Read main instruction
with open(f'prompts/qa/filter_hallucination.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

# Initialize LLMChat model
llm = LLMCompletion(args.model)

if args.eval:
    file_name = f'results/qa/filter_hallucination_eval/{args.model}/{args.answer_type}.json'
else:
    file_name = f'results/qa/filter_hallucination/{args.model}/{args.answer_type}.json'
    
directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Resume functionality
judgements = ['' for i in range(len(questions))]
if args.resume:
    try:
        with open(file_name, 'r') as f:
            judgements = json.load(f)
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")

# Process judgements
for i in tqdm(range(len(questions))):
    if judgements[i] != '':
        continue
        
    prompt = main_instruction.format(question=questions[i], answer=answers[i])
    current_output = llm(prompt)
    print(i, ':', current_output)
    judgements[i] = current_output

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with open(file_name, 'w') as f:
            json.dump(judgements, f)