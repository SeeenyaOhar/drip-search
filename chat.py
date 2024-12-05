from models.document import Document
from models.llm import LargeLanguageModel
from models.retriever import Retriever


class Chat:
    __rel_docs_n = 10
    def __init__(self, 
                 retriever: Retriever, 
                 model: LargeLanguageModel,
                 docs: list[Document]):
        self.retriever = retriever
        self.model = model
        self.docs = docs
    
    def answer_question(self, question: str) -> str:
        relevant_docs = self.retriever.get_rel_docs(n_docs=self.__rel_docs_n)
        answer = self.model.prompt(question, *relevant_docs)
        
        return answer