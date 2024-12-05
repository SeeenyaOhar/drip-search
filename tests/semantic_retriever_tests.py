import pytest
from models.semantic_retriever import SemanticRetriever
from models.document import Document

@pytest.fixture
def sample_documents():
    """
    Fixture to provide a set of sample documents.
    """
    return [
        Document(content="This document discusses artificial intelligence."),
        Document(content="This document is about machine learning and AI."),
        Document(content="This is a guide to neural networks."),
        Document(content="This text is unrelated to AI."),
        Document(content="An introduction to semantic similarity using transformers.")
    ]

@pytest.fixture
def retriever():
    """
    Fixture to provide an instance of SemanticRetriever.
    """
    return SemanticRetriever(model_name="all-mpnet-base-v2")

def test_retriever_returns_correct_number_of_docs(retriever, sample_documents):
    """
    Test that the retriever returns the correct number of documents.
    """
    prompt = "Tell me about AI and machine learning."
    n_docs = 3
    results = retriever.get_rel_docs(prompt, sample_documents, n_docs=n_docs)
    assert len(results) == n_docs, f"Expected {n_docs} documents, got {len(results)}."

def test_retriever_ranks_documents_correctly(retriever, sample_documents):
    """
    Test that the retriever ranks the most relevant document first.
    """
    prompt = "What is AI?"
    results = retriever.get_rel_docs(prompt, sample_documents, n_docs=1)
    assert len(results) == 1
    top_document = results[0]
    assert "artificial intelligence" in top_document.content.lower(), \
        "The top document does not contain the most relevant content."

def test_retriever_handles_empty_prompt(retriever, sample_documents):
    """
    Test that the retriever handles an empty prompt gracefully.
    """
    prompt = ""
    results = retriever.get_rel_docs(prompt, sample_documents, n_docs=3)
    assert len(results) == 3, "Expected 3 documents even with an empty prompt."
    assert all(isinstance(doc, Document) for doc in results), \
        "Returned results are not all Document instances."

def test_retriever_handles_empty_document_list(retriever):
    """
    Test that the retriever handles an empty document list gracefully.
    """
    prompt = "Tell me about AI."
    results = retriever.get_rel_docs(prompt, [], n_docs=3)
    assert results == [], "Expected no results for an empty document list."

def test_retriever_returns_all_docs_if_n_docs_exceeds_available(retriever, sample_documents):
    """
    Test that the retriever returns all documents if n_docs exceeds the number of available documents.
    """
    prompt = "Tell me about AI."
    n_docs = len(sample_documents) + 5  # More than available documents
    results = retriever.get_rel_docs(prompt, sample_documents, n_docs=n_docs)
    assert len(results) == len(sample_documents), \
        f"Expected {len(sample_documents)} documents, got {len(results)}."