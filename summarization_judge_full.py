import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
from architectures import LLMCompletion
from retrieve import SummaryRetriever
from utils import extract_query
from nltk.tokenize import sent_tokenize

# Argument parsing
parser = argparse.ArgumentParser(description="QA processing script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=4, help="Top K results for sentence retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--count_limit", type=int, default=10, help="Limit for the count within the loop")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/summarization_sampled_data.json', lines=True)
documents = df['document'].tolist()
summaries = df[args.answer_type + '_summary'].tolist()
retriever = SummaryRetriever(topk = args.topk)

if args.selection != None:
    add = f'_selection{args.selection}'
else:
    add = ''
    
# Read instructions
with open(f'prompts/summarization/judge_{args.form}_full{add}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

knowledge_file = f'prompts/summarization/retrieve_{args.form}{add}.txt'
with open(knowledge_file, 'r', encoding="utf-8") as f:
    knowledge_instruction = f.read()

stop_tokens = ['#Knowledge', '\n\n']
llm = LLMCompletion(args.model)
judgments = ['' for _ in range(len(documents))]

# Resume functionality
start_index = 0
file_name = f'results/summarization/judgment/{args.model}_{args.answer_type}_{args.form}_top{args.topk}'
if args.selection != None:
    file_name += f'_selection{args.selection}'
file_name += '_full.json'

if args.resume:
    try:
        with open(file_name, 'r') as f:
            judgments = json.load(f)
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")
else:
    judgments = [[] for _ in range(len(documents))]

for i in tqdm(range(len(documents))):
    if judgments[i] != []:
        continue
    sentences = sent_tokenize(summaries[i].strip())
    for k in range(0, len(sentences), 2):
        # Check if there is a pair of sentences
        if k + 1 < len(sentences):
            summary = sentences[k] + " " + sentences[k + 1]
        else:
            summary = ' '.join(sentences[k])
        summary = summaries[i].strip()
        count = 0
        prompt = main_instruction.format(summary=summary.strip())
        prompt_length = len(prompt)
        prompt += '#Thought-1#:'
        current_output = llm(prompt, stop_tokens)
        count += 1
        prompt += current_output

        while count < args.count_limit:
            if '#Judgment#' in current_output or '\n\n' in current_output:
                output = prompt[prompt_length:].strip()
                judgments[i].extend([[output, token, prob]])
                print(output)
                break
            elif current_output.endswith('#Knowledge'):
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

                knowledge = retriever.retrieve(documents[i], query)
                if args.selection != None or len(query) == 1:
                    knowledge_prompt = knowledge_instruction.format(question=query[0], knowledge=knowledge)
                else:
                    knowledge_prompt = knowledge_instruction.format(question=f'{query[0]} [{query[1]}]', knowledge=knowledge)
                knowledge_output = llm(knowledge_prompt).split('\n')[0]
                prompt += f'-{count}#:' + knowledge_output + f'\n#Thought-{count+1}#:'
            elif args.model.startswith('gpt') and '\n' == current_output[-1:]:
                try:
                    query = extract_query(current_output)
                except Exception as e:
                    print(f"Error extracting query: {e}")
                    import pdb; pdb.set_trace()
                knowledge = retriever.retrieve(documents[i], query)
                if args.selection != None or len(query) == 1:
                    knowledge_prompt = knowledge_instruction.format(question=query[0], knowledge=knowledge)
                else:
                    try:
                        knowledge_prompt = knowledge_instruction.format(question=f'{query[0]} [{query[1]}]', knowledge=knowledge)
                    except:
                        import pdb;pdb.set_trace()
                knowledge_output = llm(knowledge_prompt).split('\n')[0]
                prompt += f'#Knowledge-{count}#:' + knowledge_output + f'\n#Thought-{count+1}#:'
            else:
                output = prompt[prompt_length:].strip()
                judgments[i].extend([[output, token, prob]])
                print(output)

            current_output, token, prob = llm(prompt, stop_tokens, True)
            count += 1
            prompt += current_output
        break
    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(file_name, 'w') as f:
            json.dump(judgments, f)
