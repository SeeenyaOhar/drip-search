from models.document import Document


class Chunker:
    def chunk(self, doc: Document, desired_chunk_size: int, max_chunk_size: int) -> list[Document]:
        pass
    
class DefaultChunker(Chunker):
    def chunk(self, doc: Document, desired_chunk_size: int, max_chunk_size: int) -> list[Document]:
        """
        Splits the document content into chunks based on the desired_chunk_size,
        without exceeding the max_chunk_size.
        """
        content = doc.content
        chunks = []
        
        # Start chunking from the beginning of the content
        start = 0
        while start < len(content):
            # Determine the size of the current chunk
            chunk_end = min(start + desired_chunk_size, len(content))
            # Ensure that chunks do not exceed the max_chunk_size
            if chunk_end - start > max_chunk_size:
                chunk_end = start + max_chunk_size

            # Create a new Document object for the chunk
            chunk = Document(content[start:chunk_end])
            chunks.append(chunk)
            
            # Move the start pointer forward for the next chunk
            start = chunk_end
        
        return chunks