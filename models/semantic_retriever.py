import hashlib
import logging
import os
from sentence_transformers import SentenceTransformer, util
import torch

from models.document import Document
from models.retriever import Retriever
import numpy as np

class SemanticRetriever(Retriever):
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 logger: logging.Logger = logging.getLogger(),
                 docs=None,
                 precalc=False,
                 cache_dir='./data/embeddings'
                 ):
        self.model = SentenceTransformer(model_name)
        self.__docs = docs
        self.logger = logger
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.__docs_embeddings = self.__precalc_docs(docs) if precalc else None
        self.__precalc = precalc
        

    def set_docs(self, docs):
        self.__docs = docs
        self.__docs_embeddings = self.__precalc_docs(docs) if self.__precalc else None
    
    def get_scores(self,
                     prompt: str, 
                     n_docs=10,
                     rerank=False) -> np.ndarray:
        if len(self.__docs) == 0:
            return np.array([])
            
        if self.__docs_embeddings is None:
            self.__docs_embeddings = self.__precalc_docs(self.__docs)
        
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
        
        similarities = self.model.similarity(prompt_embedding, self.__docs_embeddings)[0]
        
        return similarities.cpu().numpy()

    def get_rel_docs(self,
                        prompt: str,
                        n_docs=10) -> list[Document]:
        sorted_indices = np.argsort(-self.get_scores(prompt, n_docs))

        top_indices = sorted_indices[:n_docs]

        return [self.__docs[idx] for idx in top_indices]
    
    def __compute_hash(self, docs):
        doc_texts = [f"{doc.content}-{doc.id}" for doc in docs]
        hash_input = "".join(doc_texts).encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    def __precalc_docs(self, docs):
        if docs is None:
            return None
        
        self.logger.debug(f"[SemanticRetriever.__precalc_docs] - precalculating {len(docs)} documents")
        
        hash = self.__compute_hash(docs)
        cache_path = os.path.join(self.cache_dir, f"{hash}.pt")
        
        if os.path.exists(cache_path):
            self.logger.debug(f"[SemanticRetriever.__precalc_docs] - loading embeddings from cache: {cache_path}")
            doc_embeddings = torch.load(cache_path)
        else:
            doc_texts = [doc.content for doc in docs]
            doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
            self.logger.debug(f"[SemanticRetriever.__precalc_docs] - finished precalculating {len(docs)} documents")
            torch.save(doc_embeddings, cache_path)
            self.logger.debug(f"[SemanticRetriever.__precalc_docs] - saved embeddings to cache: {cache_path}")
        
        return doc_embeddings