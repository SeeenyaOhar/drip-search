from models.retriever import Retriever
import logging
from models.document import Document
from rank_bm25 import BM25Okapi
import numpy as np
import spacy

class KeywordRetriever(Retriever):
    def __init__(self, logger: logging.Logger = logging.getLogger(), docs: list[Document] = None) -> None:
        self.__docs = docs
        self.__spacy_nlp = spacy.blank("en")
        self.logger = logger

    def tokenize(self, text: str):
        """Tokenize string with SpaCy. """

        tokens = self.__spacy_nlp.tokenizer(text)
        return [str(token).lower() for token in tokens]

    def tokenize_docs(self, docs: list[Document]):
        return [self.tokenize(doc.content) for doc in docs]

    def get_scores(self,
                     prompt: str,
                     n_docs=10) -> np.ndarray:
        tokenized_docs = self.tokenize_docs(self.__docs)
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_prompt = prompt.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_prompt)
        return bm25_scores

    def get_rel_docs(self, prompt, n_docs=10) -> list[Document]:
        sorted_indices = np.argsort(self.get_scores(prompt, n_docs))[::-1]
        result = [self.__docs[i] for i in sorted_indices[:n_docs]]

        return result

    def set_docs(self, docs):
        self.__docs = docs