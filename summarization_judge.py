import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
from architectures import LLMCompletion
from utils import clean_query, split_summary_into_parts, insert_newlines

# Argument parsing
parser = argparse.ArgumentParser(description="QA judgment script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=3, help="Top K results for wiki retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--query_selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/summarization_data.json', lines=True)
    
documents = df['document'].tolist()
summaries = df[args.answer_type + '_summary'].tolist()

# Load query knowledge from the stored file
if args.eval:
    file_name = f'results/summarization/query_knowledge_eval/{args.model}/{args.answer_type}_{args.form}_top{args.topk}'
else:
    file_name = f'results/summarization/query_knowledge/{args.model}/{args.answer_type}_{args.form}_top{args.topk}'
    
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
with open(f'prompts/summarization/judge_{args.form}{suffix}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

llm = LLMCompletion(args.model)

# Resume functionality
if args.resume:
    try:
        with open(judgment_file, 'r') as f:
            judgments = json.load(f)
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")
else:
    judgments = [[] for _ in range(len(documents))]

# Judgments processing
for i in tqdm(range(len(documents))):
    if judgments[i] != []:
        continue
        
    for k, summary in enumerate(split_summary_into_parts(summaries[i].strip())):
        query_knowledge = clean_query(insert_newlines(query_knowledges[i][k]))
        prompt = main_instruction.format(summary=summary, query_knowledge=query_knowledge)
        current_output = llm(prompt, return_prob=True)
        judgments[i].append(current_output)
        print(current_output[0])

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(judgment_file, 'w') as f:
            json.dump(judgments, f)