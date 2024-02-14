import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
from architectures import LLMCompletion
from utils import clean_query, insert_newlines

# Argument parsing
parser = argparse.ArgumentParser(description="QA judgment script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=2, help="Top K results for wiki retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--knowledge_type", type=str, default='ground', choices=['ground', 'wiki'], help="Type of knowledge source")
parser.add_argument("--query_selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/qa_data.json', lines=True)
questions = df['question'].tolist()
answers = df[args.answer_type + '_answer'].tolist()

# Load query knowledge from the stored file
file_name = f'results/qa/query_knowledge/{args.model}/{args.answer_type}_{args.knowledge_type}_{args.form}'
if args.knowledge_type == 'wiki':
    file_name += f'_top{args.topk}'
if args.query_selection != None:
    file_name += f'_q{args.query_selection}'
file_name += '.json'
judgment_file = file_name.replace('query_knowledge', 'judgment')

directory = os.path.dirname(judgment_file)
if not os.path.exists(directory):
    os.makedirs(directory)
    
with open(file_name, 'r') as f:
    query_knowledges = json.load(f)

if args.query_selection != None:
    suffix = f'_selection{args.query_selection}'
else:
    suffix = ''
    
# Read main instruction
with open(f'prompts/qa/judge_{args.form}{suffix}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

llm = LLMCompletion(args.model)

# Resume functionality
if args.resume:
    try:
        with open(judgment_file, 'r') as f:
            judgments = json.load(f)
    except FileNotFoundError:
        judgments = [[] for _ in range(len(questions))]
        print("No checkpoint file found, starting from scratch.")
else:
    judgments = [[] for _ in range(len(questions))]

# Judgments processing
for i in tqdm(range(len(questions))):
    if judgments[i] != []:
        continue
        
    query_knowledge = clean_query(insert_newlines(query_knowledges[i]))
    prompt = main_instruction.format(question=questions[i], answer=answers[i], query_knowledge=query_knowledge)
    current_output = llm(prompt, return_prob=True)
    judgments[i].extend(current_output)
    print(current_output[0])

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(query_knowledges) - 1:
        with open(judgment_file, 'w') as f:
            json.dump(judgments, f)