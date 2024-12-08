from models.document import Document


class Retriever:
    def get_rel_docs(self, 
                     prompt: str, 
                     n_docs=10) -> list[Document]:
        pass
    
    def set_docs(self, docs):
        pass