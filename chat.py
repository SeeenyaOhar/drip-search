import logging

from models.chunker import Chunker
from models.document import Document
from models.llm import LargeLanguageModel
from models.reranker import Reranker
from models.retriever import Retriever


class Chat:
    __retr_docs_n = 50
    __rerank_docs_n = 5
    __desired_chunk_size = 300
    __max_chunk_size = 400
    def __init__(self, 
                 retriever: Retriever, 
                 llm: LargeLanguageModel,
                 docs: list[Document],
                 chunker: Chunker,
                 reranker: Reranker,
                 logger: logging.Logger = logging.getLogger(__file__),
                 prechunk=False):
        self.retriever = retriever
        self.llm = llm
        self.docs = docs
        self.chunker = chunker
        self.reranker = reranker
        self.logger = logger
        if prechunk:
            self.__chunked_docs = self.__prechunk() 
            self.retriever.set_docs(self.__chunked_docs)
        self.__is_chunked = prechunk
        self.logger.debug(f"[Chat.__init__] prechunk={prechunk}")
    
    def __prechunk(self):
        new_docs = self.chunker.chunk_docs(self.docs, 
                                           desired_chunk_size=self.__desired_chunk_size,
                                           max_chunk_size=self.__max_chunk_size)
        self.logger.info(f"[Chat.__prechunk] - prechunked new_docs_n={len(new_docs)} total_chars={sum([len(doc.content) for doc in new_docs])}")
        self.__is_chunked = True
        
        return new_docs
    
    def answer_question(self, 
                        question: str) -> str:
        if not self.__is_chunked:
            self.__chunked_docs = self.__prechunk()
            self.retriever.set_docs(self.__chunked_docs)
        relevant_docs = self.retriever.get_rel_docs(question, 
                                                    n_docs=self.__retr_docs_n)
        reranked_docs = self.reranker.rerank(question, 
                                             relevant_docs,
                                             n_docs=self.__rerank_docs_n)
        answer = self.llm.prompt(question, *reranked_docs)
        
        return answer