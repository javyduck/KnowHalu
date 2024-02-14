import requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from FlagEmbedding import LLMEmbedder

def wiki_retrieval(queries, evi_num=2):
    if not isinstance(queries, list):
        queries = [queries]
    unique_articles = set()
    formatted_results = []
    for query in queries:
        response = requests.get(
            'http://127.0.0.1:5000/search',
            json={"query": query, "evi_num": evi_num},
        )
        if response.status_code != 200:
            raise Exception("ColBERT Search API Error: %s" % str(response))
        results = response.json()

        for r in results["passages"]:
            title, article = r.split("|", maxsplit=1)
            article_stripped = article.strip()
            if article_stripped not in unique_articles:
                unique_articles.add(article_stripped)
                formatted_results.append(f"Title: {title.strip()}. Article: {article_stripped}")
    return "\n".join(formatted_results)

class SummaryRetriever:
    def __init__(self, model_name='BAAI/bge-large-en-v1.5', topk=3):
        self.model = SentenceTransformer(model_name)
        self.instruction = 'Represent this sentence for searching relevant passages:'
        self.topk = topk
    
    @torch.no_grad()
    def retrieve(self, document, queries):
        # Split document into passages and map them to sentence indices
        passages, index_mapping = self.create_passages_with_indices(document.strip())
        
        # Embedding the passages
        p_embeddings = self.model.encode(passages, normalize_embeddings=True)

        # Embedding the queries with the instruction
        query_texts = [self.instruction + q for q in queries]
        q_embeddings = self.model.encode(query_texts, normalize_embeddings=True)

        # Compute cosine similarity scores
        scores = np.matmul(q_embeddings, p_embeddings.T)

        # Select topk passages based on sentence indices
        selected_sentence_indices = set()
        for query_scores in scores:
            top_results = np.argsort(query_scores)[-self.topk:]
            for idx in top_results:
                selected_sentence_indices.update(index_mapping[idx])

        # Sort sentence indices and construct the final text
        sorted_indices = sorted(list(selected_sentence_indices))
        sentences = sent_tokenize(document)
        return ' '.join([sentences[idx].strip() for idx in sorted_indices])

    def create_passages_with_indices(self, text, max_words = 40):
        """
        Splits the text into passages with each having up to max_words.

        :param text: A string containing the text to split.
        :param max_words: Maximum number of words allowed in a passage.
        :return: A tuple of (list of passages, index mapping for each passage).
        """
#         nltk.download('punkt')  # Ensure necessary NLTK data is downloaded
        sentences = sent_tokenize(text)
        passages = []
        index_mapping = {}
        current_passage = []
        current_word_count = 0
        current_index = 0
        for i, sentence in enumerate(sentences):
            word_count = len(word_tokenize(sentence))
            if current_word_count + word_count > max_words and current_passage:
                # Finalize the current passage and start a new one
                passages.append(' '.join(current_passage))
                index_mapping[len(passages) - 1] = list(range(current_index, i))
                current_passage = [sentence]
                current_word_count = word_count
                current_index = i
            else:
                current_passage.append(sentence)
                current_word_count += word_count

        # Add the last passage if it's not empty
        if current_passage:
            passages.append(' '.join(current_passage))
            index_mapping[len(passages) - 1] = list(range(current_index, len(sentences)))

        return passages, index_mapping
