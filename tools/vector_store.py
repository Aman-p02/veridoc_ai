import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

def get_embeddings(provider: str = "local", openai_api_key: str | None = None, google_api_key: str | None = None) -> Embeddings:
    """
    Creates and returns an embedding model instance based on the configuration.
    Falls back to local embeddings if API-based models fail or are unconfigured.
    
    Args:
        provider: 'openai', 'gemini', or 'local'.
        openai_api_key: Optional explicit API key.
        google_api_key: Optional explicit API key.
        
    Returns:
        LangChain Embeddings instance.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
    provider_clean = provider.strip().lower()
    
    if provider_clean == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and not api_key.startswith("your_"):
            try:
                from langchain_openai import OpenAIEmbeddings
                logger.info("Initializing OpenAI Embeddings...")
                return OpenAIEmbeddings()
              # Fallback if import or execution fails
            except Exception as e:
                logger.warning(f"Failed to load OpenAI Embeddings ({str(e)}). Falling back to local.")
        else:
            logger.warning("OpenAI API key missing. Falling back to local embeddings.")
            
    elif provider_clean == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key and not api_key.startswith("your_"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                logger.info("Initializing Google Gemini Embeddings...")
                return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            except Exception as e:
                logger.warning(f"Failed to load Gemini Embeddings ({str(e)}). Falling back to local.")
        else:
            logger.warning("Google API key missing. Falling back to local embeddings.")

    # Local fallback
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("Initializing local HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        # all-MiniLM-L6-v2 is an extremely efficient, fast, and high-quality open-source model
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        logger.critical(f"Critical: Failed to initialize local embeddings model: {str(e)}")
        raise e

def create_vector_store(documents: List[Document], embeddings: Embeddings) -> FAISS:
    """
    Creates an in-memory FAISS vector store from chunked documents.
    """
    if not documents:
        raise ValueError("Cannot create a vector store from an empty document list.")
    logger.info(f"Building FAISS vector index for {len(documents)} chunks...")
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info("FAISS vector store built successfully.")
    return vector_store

def query_vector_store(vector_store: FAISS, query: str, k: int = 5) -> List[Document]:
    """
    Searches the vector store for documents similar to the query.
    
    Args:
        vector_store: The FAISS vector store instance.
        query: Search string.
        k: Number of documents to retrieve.
        
    Returns:
        List of relevant LangChain Document chunks with metadata.
    """
    logger.info(f"Retrieving top {k} context chunks for query: '{query}'")
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return []
