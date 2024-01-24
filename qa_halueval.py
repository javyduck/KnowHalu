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
parser.add_argument('--knowledge_type', type=str, default='ground', help="Type of knowledge source")
parser.add_argument('--topk', type=int, default=2, help="Top K results for wiki retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/qa_sampled_data.json', lines=True)
questions = df['question'].tolist()
answers = df[args.answer_type + '_answer'].tolist()

# Load knowledge based on reasoning type
if args.knowledge_type == 'ground':
    ground_knowledge = df['knowledge'].tolist()
elif args.knowledge_type == 'wiki':
    from retrieve import wiki_retrieval
    
# Read main instruction
instruction_file = f'prompts/qa/halueval_{args.knowledge_type}.txt'
with open(instruction_file, 'r', encoding="utf-8") as f:
    main_instruction = f.read()

llm = LLMCompletion(args.model)

# Resume functionality
judgments = ['' for _ in range(len(questions))]
file_name = f'results/qa/judgment/{args.model}/halueval_{args.answer_type}_{args.knowledge_type}'
if args.knowledge_type == 'wiki':
    file_name += f'_top{args.topk}'
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

for i in tqdm(range(len(questions))):
    if judgments[i] != '':
        continue
    if args.knowledge_type in ['base', 'cot', 'simple']:
        prompt = main_instruction.format(
            question=questions[i],
            answer=answers[i],
        )
    else:
        knowledge = ground_knowledge[i] if args.knowledge_type == 'ground' else wiki_retrieval(questions[i], args.topk)
        prompt = main_instruction.format(
            question=questions[i],
            answer=answers[i],
            knowledge = knowledge
        )
    current_output = llm(prompt)
    judgments[i] = current_output
    print(current_output)

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with open(file_name, 'w') as f:
            json.dump(judgments, f)