"""
PDF Text Extraction Utilities
Extracts text from PDF files with multiple fallback methods
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")


class PDFExtractor:
    """Extract text from PDF files using multiple methods"""
    
    def __init__(self):
        self.methods = []
        if PYMUPDF_AVAILABLE:
            self.methods.append('pymupdf')
        if PDFPLUMBER_AVAILABLE:
            self.methods.append('pdfplumber')
        if PYPDF2_AVAILABLE:
            self.methods.append('pypdf2')
        
        if not self.methods:
            raise RuntimeError("No PDF extraction libraries available. Install PyMuPDF, pdfplumber, or PyPDF2")
        
        logger.info(f"Available PDF extraction methods: {', '.join(self.methods)}")
    
    def extract_text(self, pdf_path: str, method: str = 'auto') -> Dict:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('auto', 'pymupdf', 'pdfplumber', 'pypdf2')
        
        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if method == 'auto':
            # Try methods in order of preference
            for m in self.methods:
                try:
                    result = self._extract_with_method(pdf_path, m)
                    if result and result.get('text'):
                        logger.info(f"Successfully extracted text from {pdf_path.name} using {m}")
                        return result
                except Exception as e:
                    logger.warning(f"Method {m} failed for {pdf_path.name}: {e}")
                    continue
            
            raise RuntimeError(f"All extraction methods failed for {pdf_path}")
        else:
            return self._extract_with_method(pdf_path, method)
    
    def _extract_with_method(self, pdf_path: Path, method: str) -> Dict:
        """Extract text using specific method"""
        if method == 'pymupdf':
            return self._extract_pymupdf(pdf_path)
        elif method == 'pdfplumber':
            return self._extract_pdfplumber(pdf_path)
        elif method == 'pypdf2':
            return self._extract_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _extract_pymupdf(self, pdf_path: Path) -> Dict:
        """Extract text using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not available")
        
        doc = fitz.open(pdf_path)
        
        text_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_pages.append(text)
        
        full_text = "\n\n".join(text_pages)
        
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'mod_date': doc.metadata.get('modDate', ''),
        }
        
        doc.close()
        
        return {
            'text': full_text,
            'pages': text_pages,
            'page_count': len(text_pages),
            'metadata': metadata,
            'method': 'pymupdf',
            'file_name': pdf_path.name,
            'file_size': pdf_path.stat().st_size
        }
    
    def _extract_pdfplumber(self, pdf_path: Path) -> Dict:
        """Extract text using pdfplumber"""
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber not available")
        
        text_pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)
            
            metadata = pdf.metadata or {}
        
        full_text = "\n\n".join(text_pages)
        
        return {
            'text': full_text,
            'pages': text_pages,
            'page_count': len(text_pages),
            'metadata': metadata,
            'method': 'pdfplumber',
            'file_name': pdf_path.name,
            'file_size': pdf_path.stat().st_size
        }
    
    def _extract_pypdf2(self, pdf_path: Path) -> Dict:
        """Extract text using PyPDF2"""
        if not PYPDF2_AVAILABLE:
            raise RuntimeError("PyPDF2 not available")
        
        text_pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_pages.append(text)
            
            metadata = pdf_reader.metadata or {}
        
        full_text = "\n\n".join(text_pages)
        
        return {
            'text': full_text,
            'pages': text_pages,
            'page_count': len(text_pages),
            'metadata': metadata,
            'method': 'pypdf2',
            'file_name': pdf_path.name,
            'file_size': pdf_path.stat().st_size
        }
    
    def extract_from_directory(self, directory: str, pattern: str = "*.pdf") -> List[Dict]:
        """
        Extract text from all PDFs in directory
        
        Args:
            directory: Directory path
            pattern: File pattern (default: *.pdf)
        
        Returns:
            List of extraction results
        """
        directory = Path(directory)
        pdf_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.extract_text(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_file.name}: {e}")
                results.append({
                    'file_name': pdf_file.name,
                    'error': str(e),
                    'success': False
                })
        
        return results


def clean_text(text: str) -> str:
    """
    Clean extracted text
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # Remove page numbers (simple patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove headers/footers (common patterns)
    text = re.sub(r'\n[A-Z\s]+\n', '\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in last 100 chars
            last_period = text[end-100:end].rfind('.')
            if last_period != -1:
                end = end - 100 + last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def extract_metadata_from_text(text: str) -> Dict:
    """
    Extract metadata from text content
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Extract title (first line or heading)
    lines = text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            metadata['title'] = line
            break
    
    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', text[:1000])
    if year_match:
        metadata['year'] = int(year_match.group())
    
    # Extract DOI
    doi_match = re.search(r'10\.\d{4,}/[^\s]+', text[:2000])
    if doi_match:
        metadata['doi'] = doi_match.group()
    
    # Count words
    words = text.split()
    metadata['word_count'] = len(words)
    
    # Count pages (rough estimate)
    metadata['estimated_pages'] = len(text) // 2500
    
    return metadata


# Example usage
if __name__ == "__main__":
    extractor = PDFExtractor()
    
    # Test extraction
    # result = extractor.extract_text("sample.pdf")
    # print(f"Extracted {len(result['text'])} characters")
    # print(f"Metadata: {result['metadata']}")
    
    print("PDF extraction utilities loaded successfully")
    print(f"Available methods: {', '.join(extractor.methods)}")
