import os
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

def _extract_page_text(args) -> Document | None:
    page_idx, page, filename = args
    page_num = page_idx + 1
    try:
        text = page.extract_text()
        if not text or not text.strip():
            return None
        cleaned_text = text.replace('\x00', '')
        return Document(
            page_content=cleaned_text,
            metadata={
                "source": filename,
                "page": page_num,
            }
        )
    except Exception as e:
        logger.error(f"Error processing page {page_num} of {filename}: {str(e)}")
        return None

def extract_documents_from_pdf(pdf_path: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[Document]:
    """
    Extracts text page-by-page from a PDF file in parallel, cleans it, and splits it into chunks.
    Preserves page number metadata for accurate citations.
    
    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap in characters between chunks.
        
    Returns:
        List of chunked Document objects containing text and metadata.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    logger.info(f"Extracting documents from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    
    pages_to_process = [(idx, page, filename) for idx, page in enumerate(reader.pages)]
    documents = []
    
    # Process pages in parallel using ThreadPoolExecutor to prevent blocking CPU
    max_workers = min(32, multiprocessing.cpu_count() + 4)
    logger.info(f"Extracting text from {len(pages_to_process)} pages in parallel using {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_extract_page_text, pages_to_process)
        for doc in results:
            if doc is not None:
                documents.append(doc)

    if not documents:
        logger.warning(f"No extractable text found in PDF: {filename}")
        return []
        
    # Split pages into smaller chunks for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} pages into {len(chunked_docs)} text chunks.")
    return chunked_docs

def extract_pdf_sample(pdf_path: str, max_chars: int = 1500) -> str:
    """
    Extracts text from the beginning of the PDF to serve as a sample for document classification.
    """
    if not os.path.exists(pdf_path):
        return ""
        
    try:
        reader = PdfReader(pdf_path)
        sample = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                sample += text + "\n"
                if len(sample) >= max_chars:
                    break
        return sample[:max_chars].replace('\x00', '')
    except Exception as e:
        logger.error(f"Error extracting PDF sample: {str(e)}")
        return ""
