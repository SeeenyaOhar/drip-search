import logging
from sentence_transformers import SentenceTransformer, util

from models.document import Document
from models.retriever import Retriever
import numpy as np

class SemanticRetriever(Retriever):
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 logger: logging.Logger = logging.getLogger(),
                 docs=None,
                 precalc=False):
        """
        Initializes the SemanticRetriever with a SentenceTransformer model.
        :param model_name: Pre-trained model name for SentenceTransformer.
        """
        self.model = SentenceTransformer(model_name)
        self.__docs = docs
        self.logger = logger
        self.__docs_embeddings = self.__precalc_docs(docs) if precalc else None
        self.__precalc = precalc

    def set_docs(self, docs):
        self.__docs = docs
        self.__docs_embeddings = self.__precalc_docs(docs) if self.__precalc else None
    
    def get_scores(self,
                     prompt: str, 
                     n_docs=10) -> np.ndarray:
        """
        Retrieves the most relevant documents based on semantic similarity to the prompt.
        
        :param prompt: The input query string.
        :param docs: A list of Document objects to search within.
        :param n_docs: Number of top relevant documents to retrieve.
        :return: A list of top-N relevant Document objects.
        """
        if len(self.__docs) == 0:
            return []

        if self.__docs_embeddings is None:
            self.__docs_embeddings = self.__precalc_docs(self.__docs)
        
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
        
        # [0.1, 0.02, 0.5]
        similarities = self.model.similarity(prompt_embedding, self.__docs_embeddings)[0]
        return similarities.cpu().numpy()

    def get_rel_docs(self,
                        prompt: str,
                        n_docs=10) -> list[Document]:
        # Sort indices in descending order of combined scores
        sorted_indices = np.argsort(-self.get_scores(prompt, n_docs))

        # Take top n_docs
        top_indices = sorted_indices[:n_docs]

        # Return the corresponding top documents
        return [self.__docs[idx] for idx in top_indices]

    def __precalc_docs(self, docs):
        if docs is None:
            return None
        
        self.logger.debug(f"[SemanticRetriever.__precalc_docs] - precalculating {len(docs)} documents")
        doc_texts = [doc.content for doc in docs]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        self.logger.debug(f"[SemanticRetriever.__precalc_docs] - finished precalculating {len(docs)} documents")
        
        return doc_embeddings