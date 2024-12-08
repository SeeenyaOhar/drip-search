from models.document import Document
import numpy as np


class Retriever:


    def get_scores(self,
                     prompt: str, 
                     n_docs=10) -> np.ndarray:
        pass

    def get_rel_docs(self,
                        prompt: str,
                        n_docs=10) -> list[Document]:
        pass

    def set_docs(self, docs):
        pass