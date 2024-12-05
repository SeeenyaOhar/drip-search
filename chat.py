from models.chunker import Chunker
from models.document import Document
from models.llm import LargeLanguageModel
from models.retriever import Retriever


class Chat:
    __rel_docs_n = 10
    def __init__(self, 
                 retriever: Retriever, 
                 model: LargeLanguageModel,
                 docs: list[Document],
                 chunker: Chunker):
        self.retriever = retriever
        self.model = model
        self.docs = docs
        self.chunker = chunker
        self.__is_chunked = False
        
    def __prechunk(self):
        if self.__is_chunked:
            return self.docs
        
        new_docs = []
        for doc in self.docs:
            chunks = self.chunker.chunk(doc)
            new_docs.append(chunks)
        
        return new_docs
    
    def answer_question(self, 
                        question: str) -> str:
        self.docs = self.__prechunk()
        relevant_docs = self.retriever.get_rel_docs(question, 
                                                    n_docs=self.__rel_docs_n)
        answer = self.model.prompt(question, *relevant_docs)
        
        return answer