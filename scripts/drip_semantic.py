import logging.handlers
from models.document import Document
from models.groqllm import GroqModel
from models.semantic_retriever import SemanticRetriever
import os
import logging

def chunk_documents(document, chunk_size=500):
    """
    Splits the document content into smaller chunks based on the given chunk size.
    """
    content = document.content
    return [Document(content[i:i + chunk_size]) for i in range(0, len(content), chunk_size)]

if __name__ == '__main__':
    logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('logs/app.log', mode='w')])
    logging.info('[main] Starting load_dotenv')
    from dotenv import load_dotenv
    load_dotenv()
    logging.info('[main] Finished os_dotenv')
    retriever = SemanticRetriever()
    documents = []
    
    for item in sorted(os.listdir('data')):
        file_path = os.path.join('data', item)
        
        if os.path.isfile(file_path) and item.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            
            document = Document(content)
            
            chunks = chunk_documents(document, chunk_size=500)
            documents.extend(chunks) 
    logging.info(f'[main] Chunked the documents, documents_len={len(documents)}')
    groqllm = GroqModel()
    test_prompt = 'What are the total sales of Carhartt in 1990?'
    context = retriever.get_rel_docs(test_prompt, documents, n_docs=3)
    logging.info(f'[main] Found rel docs, context_len={len(context)}')
    logging.info(f'[main] --- GROQ LLM Answer: {groqllm.prompt(test_prompt, context)}')
