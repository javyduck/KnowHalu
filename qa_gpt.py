import os
import time
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="QA processing script.")
parser.add_argument("--model", type=str, default='gpt-35-turbo', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=2, help="Top K results for wiki retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--knowledge_type", type=str, default='ground', choices=['ground', 'wiki'], help="Type of knowledge source")
parser.add_argument("--query_selection", type=int, default=None, help="Index for the query to use")
parser.add_argument("--answer_selection", type=int, default=None, help="Index for the query to answer")
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints")
parser.add_argument("--count_limit", type=int, default=10, help="Limit for the count within the loop")
args = parser.parse_args()

def run_command():
    # Replace this with your command
#     command = f'python qa_query.py --model {args.model} --form {args.form} --topk {args.topk} ' \
#               f'--answer_type {args.answer_type} --knowledge_type {args.knowledge_type} ' \
#               f'--save_freq {args.save_freq} --resume --eval'

    command = "python qa_halueval.py --model gpt-35-turbo --knowledge_type wiki --answer_type right --topk 2 --resume && python qa_halueval.py --model gpt-35-turbo --knowledge_type wiki --answer_type hallucinated --topk 2 --resume"
    return os.system(command)

# Infinite loop to keep retrying the command after any error
while True:
    status = run_command()

    # If status is non-zero, it indicates an error
    if status != 0:
        print("Error encountered. Waiting for 20 seconds before retrying...")
        time.sleep(20)
    else:
        break
