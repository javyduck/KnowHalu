# KnowHalu: Hallucination Detection via Multi-Form Knowledge Based Factual Checking 🕵️‍♂️📚

## Overview

KnowHalu introduces a pioneering approach to hallucination detection in AI-generated content, focusing on enhancing the reliability and factual accuracy of language models 🧠. Our work is structured around a comprehensive pipeline designed to identify and rectify hallucinations through a multi-stage factual checking process 🛠️.

The hallucination detection process begins with **Non-Fabrication Hallucination Checking**, a preliminary phase aimed at identifying hallucinations based on the specificity of the answers provided 🕵️. This is followed by a detailed **Factual Checking** procedure, comprising five critical steps:

1. **Step-wise Reasoning and Query**: Decomposes the initial query into smaller, manageable sub-queries, facilitating a more granular factual verification 🔍.
2. **Knowledge Retrieval**: Employs Retrieval-Augmented Generation (RAG) for unstructured knowledge and extracts structured knowledge in the form of triplets for each sub-query 📖.
3. **Knowledge Optimization**: Utilizes Large Language Models (LLMs) to summarize and refine the retrieved knowledge into different forms, optimizing it for further processing 🔄.
4. **Judgment Based on Multi-form Knowledge**: Applies LLMs to critically assess the answers to sub-queries, utilizing the optimized knowledge forms 🧐.
5. **Aggregation**: Enhances the final judgment by amalgamating predictions from different forms of knowledge, ensuring a thorough and nuanced evaluation 📊.

## Environment Setup

To set up the environment for running KnowHalu, ensure you have Conda installed. Follow these steps to create a Conda environment with all required dependencies:

```
bashCopy code
conda create --name knowhalu python=3.8
conda activate knowhalu
pip install -r requirements.txt
```

This setup guarantees that you have all the necessary libraries and frameworks to execute KnowHalu's pipeline effectively.

## Usage

### QA Task Hallucination Detection

For detecting hallucinations in QA tasks, we provide `qa_relevance.py` for Non-Fabrication Hallucination Checking and `qa_query.py` for gathering queries and related knowledge. Use the following parameters for detailed customization:

```
bashCopy code
python qa_query.py --model Starling-LM-7B-alpha --form semantic --topk 2 --answer_type right --knowledge_type ground --query_selection None
```

- `--model`: Specifies the model to be used (default: 'Starling-LM-7B-alpha').
- `--form`: Determines the form of the data, either structured or unstructured (default: 'semantic').
- `--topk`: Sets the number of top results to retrieve from Wikipedia (default: 2).
- `--answer_type`: Defines the type of answer, either 'right' or 'hallucinated'.
- `--knowledge_type`: Indicates the source of knowledge, either 'ground' (off-the-shelf) or 'wiki' (retrieved).
- `--query_selection`: Specifies the index for the query formulation used; 0 for specific, 1 for general, and None for using both.

Final judgments are obtained using `qa_judge.py`, following the collection of queries and knowledge.

### Text Summarization Task Hallucination Detection

The process for detecting hallucinations in text summarization tasks involves `summarization_query.py` for initial query and knowledge collection, followed by `summarization_judge.py` for final judgment. The usage is similar to the QA task, adapted for summarization purposes.

This comprehensive guide should enable you to effectively utilize KnowHalu for detecting and analyzing hallucinations in AI-generated content, ensuring higher factual accuracy and reliability 🌟.