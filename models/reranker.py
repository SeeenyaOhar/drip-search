import logging

import numpy as np
from models.document import Document
from sentence_transformers import CrossEncoder, util



class Reranker:
    def rerank(self, 
               docs: list[Document], 
               n_docs: int = 10):
        pass
    
    
class CrossEncoderReranker:
    def __init__(self, 
                 model_name='cross-encoder/ms-marco-MiniLM-L-12-v2',
                 max_length=512,
                 logger=logging.getLogger(__file__)):
        self.model = CrossEncoder(model_name, 
                                  max_length=max_length)
        self.logger = logger
        
    def rerank(self, 
               prompt: str,
               docs: list[Document], 
               n_docs: int = 10):
        scores = self.model.predict(
            [(prompt, doc.content) for doc in docs]
        )
        sorted_indices = np.argsort(-scores)

        top_indices = sorted_indices[:n_docs]
        self.logger.info(f"[CrossEncoderReranker.rerank] - reranked to {n_docs} documents")
        return [docs[idx] for idx in top_indices]
        
        
        
    