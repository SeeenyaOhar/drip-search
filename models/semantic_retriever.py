from sentence_transformers import SentenceTransformer, util
from models.document import Document
from models.retriever import Retriever

class SemanticRetriever(Retriever):
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initializes the SemanticRetriever with a SentenceTransformer model.
        :param model_name: Pre-trained model name for SentenceTransformer.
        """
        self.model = SentenceTransformer(model_name)
    
    def get_rel_docs(self, 
                     prompt: str, 
                     docs: list[Document], 
                     n_docs=10) -> list[Document]:
        """
        Retrieves the most relevant documents based on semantic similarity to the prompt.
        
        :param prompt: The input query string.
        :param docs: A list of Document objects to search within.
        :param n_docs: Number of top relevant documents to retrieve.
        :return: A list of top-N relevant Document objects.
        """
        if len(docs) == 0:
            return []
        doc_texts = [doc.content for doc in docs]
        
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        
        similarities = self.model.similarity(prompt_embedding, doc_embeddings)[0]
        top_indices = similarities.topk(k=min(n_docs, len(docs))).indices
        
        return [docs[idx] for idx in top_indices.tolist()]