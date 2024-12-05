from models.document import Document
from models.groqllm import GroqModel
from models.semantic_retriever import SemanticRetriever
import os

def chunk_documents(document, chunk_size=500):
    """
    Splits the document content into smaller chunks based on the given chunk size.
    """
    content = document.content
    return [Document(content[i:i + chunk_size]) for i in range(0, len(content), chunk_size)]

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    retriever = SemanticRetriever()
    documents = []
    
    for item in sorted(os.listdir('data')):
        file_path = os.path.join('data', item)
        
        if os.path.isfile(file_path) and item.endswith('.txt'):
            # Open the .txt file and read the content
            with open(file_path, 'r') as file:
                content = file.read()
            
            document = Document(content)
            
            chunks = chunk_documents(document, chunk_size=500)  # Adjust chunk_size as needed
            documents.extend(chunks)  # Add the chunks to the documents list
    groqllm = GroqModel()
    test_prompt = 'What are the total sales of Carhartt in 1990?'
    context = retriever.get_rel_docs(test_prompt, documents, n_docs=3)
    print(groqllm.prompt(test_prompt, context))
