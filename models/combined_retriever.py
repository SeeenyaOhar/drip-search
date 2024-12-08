from models.document import Document
from models.keyword_retriever import KeywordRetriever
from models.retriever import Retriever
from models.semantic_retriever import SemanticRetriever
import numpy as np


class CombinedRetriever(Retriever):
    def __init__(self, documents: list[Document], semantic: SemanticRetriever, keyword: KeywordRetriever):
        self.__semantic = semantic
        self.__keyword = keyword
        self.__documents = documents

    def get_rel_docs(self, prompt: str, n_docs=10) -> list[Document]:
        if not self.__semantic:
            return self.__keyword.get_rel_docs(prompt, n_docs)
        elif not self.__keyword:
            return self.__semantic.get_rel_docs(prompt, n_docs)

        semantic_scores = self.__semantic.get_scores(prompt, n_docs)
        keyword_scores = self.__keyword.get_scores(prompt, n_docs)

        combined_scores = 0.7 * semantic_scores + 0.3 * keyword_scores
        
        sorted_indices = np.argsort(-combined_scores)
        top_indices = sorted_indices[:n_docs]

        return [self.__documents[idx] for idx in top_indices]

    def set_docs(self, docs):
        self.__semantic.set_docs(docs) if self.__semantic else None
        self.__keyword.set_docs(docs) if self.__keyword else None
        self.__documents = docs



