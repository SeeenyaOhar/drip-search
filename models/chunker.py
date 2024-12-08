import json
import os
from models.document import Document
import numpy as np
import logging


class Chunker:
    def chunk(self, doc: Document, desired_chunk_size: int, max_chunk_size: int) -> list[Document]:
        pass
    
    def chunk_docs(self, docs: list[Document], desired_chunk_size, max_chunk_size: int) -> list[Document]:
        pass
    
class DefaultChunker(Chunker):
    def __init__(self, 
                 logger: logging.Logger,
                 save_on_chunk: bool = False,
                 chunk_save_dir: str = None):
        self.logger = logger
        self.save_on_chunk = save_on_chunk
        self.chunk_save_dir = chunk_save_dir
        
        if save_on_chunk and not chunk_save_dir:
            raise ValueError("chunk_save_dir must be specified if save_on_chunk is True")
        if chunk_save_dir:
            os.makedirs(chunk_save_dir, exist_ok=True)
            
    def __get_chunk_id(self, 
                       doc: Document, 
                       chunk_n: int):
        return f"{doc.id}-{chunk_n}"
    
    def __save_chunk(self, original_doc: Document, chunks: list[Document]):
        """
        Save the chunks to files in chunk_save_dir using the original doc's ID.
        """
        doc_id = original_doc.id
        if doc_id is None:
            return
        save_path = os.path.join(self.chunk_save_dir, f"{doc_id}.json")
        data_to_save = [chunk.content for chunk in chunks]
        
        with open(save_path, "w") as file:
            json.dump(data_to_save, file)
        
        self.logger.debug(f"[__save_chunk] - Chunks saved for doc_id={doc_id} at {save_path}")
    
    def __retrieve_chunks(self, original_doc: Document) -> list[Document]:
        """
        Retrieve the saved chunks for a document by its ID.
        """
        doc_id = original_doc.id
        save_path = os.path.join(self.chunk_save_dir, f"{doc_id}.json")
        
        if not os.path.exists(save_path):
            self.logger.warning(f"[__retrieve_chunks] - No saved chunks for doc_id={doc_id}")
            return []
        
        with open(save_path, "r") as file:
            chunk_data = json.load(file)
        
        self.logger.debug(f"[__retrieve_chunks] - Retrieved {len(chunk_data)} chunks for doc_id={doc_id}")
        return [Document(content=chunk, id=self.__get_chunk_id(original_doc, i)) for i,chunk in enumerate(chunk_data)]
    
    def __is_saved(self, doc: Document) -> bool:
        """
        Check if chunks for a document are already saved.
        """
        doc_id = doc.id
        if doc_id is None:
            return False
        save_path = os.path.join(self.chunk_save_dir, f"{doc_id}.json")
        return os.path.exists(save_path)
        
    def chunk(self, doc: Document, desired_chunk_size: int, max_chunk_size: int) -> list[Document]:
        """
        Splits the document content into chunks based on the desired_chunk_size,
        without exceeding the max_chunk_size.
        """
        if self.__is_saved(doc):
            self.__retrieve_chunks(doc)
        
        content = doc.content
        chunks = []
        
        start = 0
        chunk_n = 0
        while start < len(content):
            chunk_end = min(start + desired_chunk_size, len(content))
            if chunk_end - start > max_chunk_size:
                chunk_end = start + max_chunk_size

            chunk = Document(content=content[start:chunk_end],
                             id=self.__get_chunk_id(doc, chunk_n))
            chunks.append(chunk)
            
            start = chunk_end
            chunk_n += 1
        
        if not self.__is_saved(doc) and self.save_on_chunk:
            self.__save_chunk(doc, chunks)
        
        return chunks
    
    def chunk_docs(self, docs: list[Document], desired_chunk_size, max_chunk_size: int) -> list[Document]:
        new_docs = []
        for doc in docs:
            chunks = self.chunk(doc, 
                                        desired_chunk_size=desired_chunk_size, 
                                        max_chunk_size=max_chunk_size)
            new_docs.extend(chunks)
        self.logger.debug(f"[chunk_docs] - avg_size={np.mean([len(doc.content) for doc in new_docs])}")
        return new_docs