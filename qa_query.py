import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
from architectures import LLMCompletion
from utils import extract_query

# Argument parsing
parser = argparse.ArgumentParser(description="QA processing script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=2, help="Top K results for wiki retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--knowledge_type", type=str, default='ground', choices=['ground', 'wiki'], help="Type of knowledge source")
parser.add_argument("--query_selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--count_limit", type=int, default=10, help="Limit for the count within the loop")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/qa_data.json', lines=True)
    
questions = df['question'].tolist()
answers = df[args.answer_type + '_answer'].tolist()

# Read instructions
if args.query_selection != None:
    suffix = f'_selection{args.query_selection}'
else:
    suffix = ''
with open(f'prompts/qa/query_{args.form}{suffix}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

knowledge_file = f'prompts/qa/retrieve_{args.knowledge_type}_{args.form}{suffix}.txt'
with open(knowledge_file, 'r', encoding="utf-8") as f:
    knowledge_instruction = f.read()
    
stop_tokens = ['#Knowledge', '\n\n']
llm = LLMCompletion(args.model)

if args.knowledge_type == 'ground':
    ground_knowledge = df['knowledge'].tolist()
else:
    from retrieve import wiki_retrieval
    
# Resume functionality
file_name = f'results/qa/query_knowledge/{args.model}/{args.answer_type}_{args.knowledge_type}_{args.form}'
if args.knowledge_type == 'wiki':
    file_name += f'_top{args.topk}'
if args.query_selection != None:
    file_name += f'_q{args.query_selection}'
file_name += '.json'

directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
if args.resume:
    try:
        with open(file_name, 'r') as f:
            query_knowledge = json.load(f)
    except:
        query_knowledge = ['' for _ in range(len(questions))]
        print("No checkpoint file found, starting from scratch.")
else:
    query_knowledge = ['' for _ in range(len(questions))]

for i in tqdm(range(len(questions))):
    if query_knowledge[i] != '':
        continue
    count = 0
    prompt = main_instruction.format(question=questions[i], answer=answers[i])
    prompt_length = len(prompt)
    prompt += '#Thought-1#:'
    current_output = llm(prompt, stop_tokens)
    count += 1
    if args.model.startswith('gpt'):
        prompt += ' ' + current_output
    else:
        prompt += current_output
    while count < args.count_limit:
        if '\n\n' in current_output:
            output = prompt[prompt_length:].strip()
            query_knowledge[i] = output
            print(output)
            break
        elif current_output.endswith('#Knowledge') or (args.model.startswith('gpt') and '\n' == current_output[-1:]) or ('Query-' in  current_output.split('\n')[-1]) :
            if 'Query-' in  current_output.split('\n')[-1]:
                current_output += '\n'
            query = extract_query(current_output)
            if len(query) == 0:
                last_newline_index = prompt.rfind('\n')
                prompt = prompt[:last_newline_index]
                prompt += f'\n#Query-{count}#:'
                current_output = llm(prompt, stop_tokens)
                prompt += current_output
                query = extract_query(f'#Query-{count}#:' + current_output)
                if len(query) == 0:
                    import pdb; pdb.set_trace()
            
            knowledge = ground_knowledge[i] if args.knowledge_type == 'ground' else wiki_retrieval(query, args.topk)
            if args.query_selection != None or len(query) < 2:
                knowledge_prompt = knowledge_instruction.format(question=query[0], knowledge=knowledge)
            else:
                knowledge_prompt = knowledge_instruction.format(question=f'{query[0]} [{query[1]}]', knowledge=knowledge)
            knowledge_output = llm(knowledge_prompt).split('\n')[0]
            if args.model.startswith('gpt'):
                if not prompt.endswith('\n'):
                    prompt += '\n'
                prompt += f'#Knowledge-{count}#: ' + knowledge_output + f'\n#Thought-{count+1}#:'
            else:
                if not prompt.endswith('\n#Knowledge'):
                    prompt += '\n#Knowledge'
                prompt += f'-{count}#:' + knowledge_output + f'\n#Thought-{count+1}#:'
        else:
            output = prompt[prompt_length:].strip()
            query_knowledge[i] = output
            print(output)
            break

        current_output = llm(prompt, stop_tokens)
        count += 1
        if args.model.startswith('gpt'):
            prompt += ' ' + current_output
        else:
            prompt += current_output

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with open(file_name, 'w') as f:
            json.dump(query_knowledge, f)
