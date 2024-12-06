import logging
from sentence_transformers import SentenceTransformer, util

from models.document import Document
from models.retriever import Retriever

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
    
    def get_rel_docs(self, 
                     prompt: str, 
                     n_docs=10) -> list[Document]:
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
        
        similarities = self.model.similarity(prompt_embedding, self.__docs_embeddings)[0]
        top_indices = similarities.topk(k=min(n_docs, len(self.__docs))).indices
        
        rel_docs = [self.__docs[idx] for idx in top_indices.tolist()]
        self.logger.info(f"[SemanticRetriever.get_rel_docs] prompt={prompt} docs_n={len(self.__docs)} rel_docs_n={len(rel_docs)} rel_docs_total_chars={sum([len(doc.content) for doc in rel_docs])}")
        
        return rel_docs
    
    def __precalc_docs(self, docs):
        if docs is None:
            return None
        
        self.logger.debug(f"[SemanticRetriever.__precalc_docs] - precalculating {len(docs)} documents")
        doc_texts = [doc.content for doc in docs]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        self.logger.debug(f"[SemanticRetriever.__precalc_docs] - finished precalculating {len(docs)} documents")
        
        return doc_embeddings