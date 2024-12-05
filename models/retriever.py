from models.document import Document


class Retriever:
    def get_rel_docs(self, 
                     prompt: str, 
                     docs: list[Document], 
                     n_docs=10) -> list[Document]:
        pass