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
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/summarization_sampled_data.json', lines=True)
documents = df['document'].tolist()
summaries = df[args.answer_type + '_summary'].tolist()

# Read main instruction
instruction_file = f'prompts/summarization/halueval_{args.knowledge_type}.txt'
with open(instruction_file, 'r', encoding="utf-8") as f:
    main_instruction = f.read()

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
        print("No checkpoint file found, starting from scratch.")

for i in tqdm(range(len(documents))):
    if judgments[i] != '':
        continue
    prompt = main_instruction.format(
        document=documents[i],
        summary=summaries[i],
    )
    current_output = llm(prompt)
    judgments[i] = current_output
    print(current_output)

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(file_name, 'w') as f:
            json.dump(judgments, f)