## Installation

Before running any scripts, ensure you have all the necessary packages installed. Run the following command to install dependencies:

```
pip install -r requirements.txt
```

## Baseline Evaluations

To evaluate the baseline performance of language models, you can use the following commands. These scripts assess models like GPT-3.5 Turbo 1106 and GPT-4 on various knowledge and answer types:

```
python qa_halueval.py --model gpt-3.5-turbo-1106 --knowledge_type ground --answer_type right
python qa_halueval.py --model gpt-3.5-turbo-1106 --knowledge_type ground --answer_type hallucinated
python summarization_halueval.py --model gpt-3.5-turbo-1106 --knowledge_type cot --answer_type right
python summarization_halueval.py --model gpt-3.5-turbo-1106 --knowledge_type cot --answer_type hallucinated
python qa_halueval.py --model gpt-4-1106-preview --knowledge_type simple --answer_type right
python qa_halueval.py --model gpt-4-1106-preview --knowledge_type simple --answer_type hallucinated
```

## Our Evaluation Approach

Our approach involves a two-step process: querying and judging the relevance of the responses. This method is applied to various models to assess their performance accurately.

```
python qa_relevance.py --model gpt-3.5-turbo-1106 --answer_type right
python qa_relevance.py --model gpt-3.5-turbo-1106 --answer_type hallucinated
python qa_query.py --model gpt-3.5-turbo-1106 --answer_type right --form semantic && python qa_judge.py --model gpt-3.5-turbo-1106 --answer_type right --form semantic
python qa_query.py --model gpt-3.5-turbo-1106 --answer_type right --form triplet && python qa_judge.py --model gpt-3.5-turbo-1106 --answer_type right --form triplet
python qa_query.py --model gpt-3.5-turbo-1106 --answer_type hallucinated --form semantic && python qa_judge.py --model gpt-3.5-turbo-1106 --answer_type hallucinated --form semantic
python qa_query.py --model gpt-3.5-turbo-1106 --answer_type hallucinated --form triplet && python qa_judge.py --model gpt-3.5-turbo-1106 --answer_type hallucinated --form triplet
```

### Resuming Evaluations

To resume an evaluation from the last checkpoint, simply add `--resume` to any of the above commands.