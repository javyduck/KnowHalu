import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
from architectures import LLMCompletion
from utils import extract_query, split_summary_into_parts
from retrieve import SummaryRetriever

# Argument parsing
parser = argparse.ArgumentParser(description="Summary processing script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=3, help="Top K results for document retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--query_selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--answer_selection", type=int, default=None, help="Index for the query to answer")
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints")
parser.add_argument("--count_limit", type=int, default=10, help="Limit for the count within the loop")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
parser.add_argument("--eval", action="store_true", help="Use Eval data")
args = parser.parse_args()

if args.eval:
    df = pd.read_json('data/summarization_sampled_data_val.json', lines=True)
else:
    df = pd.read_json('data/summarization_sampled_data.json', lines=True)
    
documents = df['document'].tolist()
summaries = df[args.answer_type + '_summary'].tolist()
retriever = SummaryRetriever(topk = args.topk)

# Read instructions
if args.query_selection != None:
    suffix = f'_selection{args.query_selection}'
    args.answer_selection = args.query_selection
else:
    suffix = ''
with open(f'prompts/summarization/query_{args.form}{suffix}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

if args.answer_selection != None:
    suffix = f'_selection{args.answer_selection}'
else:
    suffix = ''
knowledge_file = f'prompts/summarization/retrieve_{args.form}{suffix}.txt'
with open(knowledge_file, 'r', encoding="utf-8") as f:
    knowledge_instruction = f.read()
######################
    
stop_tokens = ['#Knowledge', '\n\n']
llm = LLMCompletion(args.model)

# Resume functionality
if args.eval:
    file_name = f'results/summarization/query_knowledge_eval/{args.model}/{args.answer_type}_{args.form}_top{args.topk}'
else:
    file_name = f'results/summarization/query_knowledge/{args.model}/{args.answer_type}_{args.form}_top{args.topk}'
    
if args.query_selection != None:
    file_name += f'_q{args.query_selection}'
if args.answer_selection != None:
    file_name += f'_a{args.answer_selection}'
file_name += '.json'

directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
if args.resume:
    try:
        with open(file_name, 'r') as f:
            query_knowledge = json.load(f)
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")
else:
    query_knowledge = [[] for _ in range(len(documents))]

for i in tqdm(range(len(documents))):
    if query_knowledge[i] != []:
        continue
        
    for summary in split_summary_into_parts(summaries[i].strip()):
        count = 0
        prompt = main_instruction.format(summary=summary)
        prompt_length = len(prompt)
        prompt += '#Thought-1#:'
        current_output = llm(prompt, stop_tokens)
        last_output = current_output
        count += 1
        if args.model.startswith('gpt'):
            prompt += ' ' + current_output
        else:
            prompt += current_output

        while count < args.count_limit:
            if '\n\n' in current_output or '#Done#' in current_output or 'further queries' in current_output:
                break
            elif current_output.endswith('#Knowledge') or (args.model.startswith('gpt') and '\n' == current_output[-1:]) or ('Query-' in  current_output.split('\n')[-1]) or (args.model.startswith('gpt') and 'Query-' not in current_output):
                if 'Query-' in  current_output.split('\n')[-1]:
                    current_output += '\n'
                elif 'Query-' not in current_output and not current_output.endswith('\n'):
                    prompt += '\n'
                query = extract_query(current_output)
                if len(query) == 0:
                    last_newline_index = prompt.rfind('\n')
                    prompt = prompt[:last_newline_index]
                    prompt += f'\n#Query-{count}#:'
                    current_output = llm(prompt, stop_tokens).split('\n')[0]
                    if args.model.startswith('gpt'):
                        prompt += ' ' + current_output + '\n'
                    else:
                        prompt += current_output + '\n'
                    query = extract_query(f'#Query-{count}#:' + current_output)
                    if len(query) == 0:
                        import pdb; pdb.set_trace()

                knowledge = retriever.retrieve(documents[i], query)
                if args.query_selection != None or len(query) < 2:
                    knowledge_prompt = knowledge_instruction.format(question=query[0], knowledge=knowledge)
                elif args.answer_selection == None:
                    knowledge_prompt = knowledge_instruction.format(question=f'{query[0]} [{query[1]}]', knowledge=knowledge)
                elif args.answer_selection != None:
                    knowledge_prompt = knowledge_instruction.format(question=f'{query[args.answer_selection]}]', knowledge=knowledge)
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
                break

            current_output = llm(prompt, stop_tokens)
            if current_output == last_output:
                break
            else:
                last_output = current_output
                
            count += 1
            if args.model.startswith('gpt'):
                prompt += ' ' + current_output
            else:
                prompt += current_output
                
        output = prompt[prompt_length:].strip()
        query_knowledge[i].append(output)
        print(output)
        
    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(file_name, 'w') as f:
            json.dump(query_knowledge, f)
