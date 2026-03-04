"""
Text Processing Utilities
Clean, chunk, and process extracted text
"""

import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean extracted text from PDFs
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Remove page numbers (simple patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\n- \d+ -\n', '\n', text)
    
    # Remove common headers/footers
    text = re.sub(r'\n[A-Z\s]{10,}\n', '\n', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:!?()\[\]{}\-\'\"°%$€£¥/\\]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def chunk_text_fixed(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into fixed-size overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
    
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
            search_text = text[max(start, end-100):end]
            last_period = search_text.rfind('.')
            last_newline = search_text.rfind('\n')
            
            break_point = max(last_period, last_newline)
            if break_point != -1:
                end = max(start, end-100) + break_point + 1
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:  # Minimum chunk size
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def chunk_text_semantic(text: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Split text into semantic chunks (by paragraphs/sections)
    
    Args:
        text: Input text
        max_chunk_size: Maximum chunk size in characters
    
    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If single paragraph is too large, split it
        if para_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            temp_chunk = []
            temp_size = 0
            
            for sent in sentences:
                if temp_size + len(sent) > max_chunk_size and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [sent]
                    temp_size = len(sent)
                else:
                    temp_chunk.append(sent)
                    temp_size += len(sent)
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        
        # Add paragraph to current chunk
        elif current_size + para_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract sections from text based on headings
    
    Args:
        text: Input text
    
    Returns:
        Dictionary mapping section titles to content
    """
    sections = {}
    
    # Common heading patterns
    heading_patterns = [
        r'^([A-Z][A-Za-z\s]{3,50})\n',  # Title case headings
        r'^\d+\.\s+([A-Za-z\s]{3,50})\n',  # Numbered headings
        r'^([A-Z\s]{5,50})\n',  # All caps headings
    ]
    
    current_section = "Introduction"
    current_content = []
    
    lines = text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if line is a heading
        is_heading = False
        for pattern in heading_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = match.group(1).strip()
                current_content = []
                is_heading = True
                break
        
        if not is_heading and line_stripped:
            current_content.append(line_stripped)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def extract_key_terms(text: str) -> List[str]:
    """
    Extract key technical terms from text
    
    Args:
        text: Input text
    
    Returns:
        List of key terms
    """
    # Glass-specific terms
    glass_terms = [
        'glass', 'silica', 'soda-lime', 'borosilicate', 'tempered', 'annealed',
        'float glass', 'viscosity', 'transition temperature', 'melting', 'forming',
        'annealing', 'tempering', 'coating', 'laminated', 'toughened', 'refractive index',
        'density', 'thermal expansion', 'chemical durability', 'optical properties',
        'mechanical properties', 'glass-ceramic', 'crystallization', 'nucleation',
        'furnace', 'batch', 'cullet', 'refining', 'homogenization', 'fining',
        'defects', 'bubbles', 'stones', 'cord', 'stress', 'birefringence'
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in glass_terms:
        if term in text_lower:
            found_terms.append(term)
    
    # Extract chemical formulas
    formula_pattern = r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b'
    formulas = re.findall(formula_pattern, text)
    
    # Filter for common glass oxides
    oxide_patterns = ['SiO2', 'Na2O', 'CaO', 'MgO', 'Al2O3', 'K2O', 'B2O3', 'Fe2O3', 'PbO']
    for formula in formulas:
        if formula in oxide_patterns:
            found_terms.append(formula)
    
    return list(set(found_terms))


def categorize_text(text: str) -> List[str]:
    """
    Categorize text content by topic
    
    Args:
        text: Input text
    
    Returns:
        List of categories
    """
    categories = []
    text_lower = text.lower()
    
    category_keywords = {
        'composition': ['composition', 'chemical', 'oxide', 'batch', 'formula', 'sio2', 'na2o'],
        'properties': ['property', 'properties', 'density', 'viscosity', 'refractive', 'thermal', 'mechanical'],
        'processing': ['processing', 'melting', 'forming', 'annealing', 'tempering', 'manufacturing'],
        'defects': ['defect', 'defects', 'bubble', 'stone', 'cord', 'stress', 'crack'],
        'applications': ['application', 'applications', 'architectural', 'automotive', 'container', 'optical'],
        'fundamentals': ['structure', 'theory', 'bonding', 'network', 'glass formation'],
        'standards': ['standard', 'astm', 'iso', 'specification', 'requirement', 'test method'],
        'coatings': ['coating', 'coatings', 'thin film', 'surface', 'deposition'],
        'glass_types': ['soda-lime', 'borosilicate', 'lead', 'aluminosilicate', 'glass-ceramic'],
    }
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                categories.append(category)
                break
    
    return list(set(categories))


def extract_metadata_from_text(text: str) -> Dict:
    """
    Extract metadata from text content
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Extract title (first meaningful line)
    lines = text.split('\n')
    for line in lines[:20]:
        line = line.strip()
        if 10 < len(line) < 200 and not line.isdigit():
            metadata['title'] = line
            break
    
    # Extract year
    year_matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
    if year_matches:
        years = [int(y) for y in year_matches]
        metadata['year'] = max(years)  # Use most recent year
    
    # Extract DOI
    doi_match = re.search(r'10\.\d{4,}/[^\s]+', text[:3000])
    if doi_match:
        metadata['doi'] = doi_match.group()
    
    # Extract authors (simple pattern)
    author_pattern = r'(?:by|author[s]?:?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    author_match = re.search(author_pattern, text[:2000], re.IGNORECASE)
    if author_match:
        metadata['author'] = author_match.group(1)
    
    # Count statistics
    words = text.split()
    metadata['word_count'] = len(words)
    metadata['char_count'] = len(text)
    metadata['estimated_pages'] = len(text) // 2500
    
    # Extract key terms
    metadata['key_terms'] = extract_key_terms(text)
    
    # Categorize content
    metadata['categories'] = categorize_text(text)
    
    return metadata


def deduplicate_chunks(chunks: List[str], similarity_threshold: float = 0.9) -> List[str]:
    """
    Remove duplicate or highly similar chunks
    
    Args:
        chunks: List of text chunks
        similarity_threshold: Similarity threshold (0-1)
    
    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return []
    
    unique_chunks = []
    seen_hashes = set()
    
    for chunk in chunks:
        # Simple hash-based deduplication
        chunk_hash = hash(chunk.strip().lower())
        
        if chunk_hash not in seen_hashes:
            unique_chunks.append(chunk)
            seen_hashes.add(chunk_hash)
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
    
    return unique_chunks


def validate_text_quality(text: str) -> Dict:
    """
    Validate text quality
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'length': len(text),
        'word_count': len(text.split()),
        'has_content': len(text.strip()) > 100,
        'avg_word_length': 0,
        'sentence_count': 0,
        'quality_score': 0.0
    }
    
    if not text.strip():
        return metrics
    
    # Calculate average word length
    words = text.split()
    if words:
        metrics['avg_word_length'] = sum(len(w) for w in words) / len(words)
    
    # Count sentences
    sentences = re.split(r'[.!?]+', text)
    metrics['sentence_count'] = len([s for s in sentences if s.strip()])
    
    # Calculate quality score
    score = 0.0
    
    # Length check
    if 100 < len(text) < 100000:
        score += 0.3
    
    # Word length check (should be reasonable)
    if 3 < metrics['avg_word_length'] < 10:
        score += 0.2
    
    # Sentence structure check
    if metrics['sentence_count'] > 5:
        score += 0.2
    
    # Contains glass-related terms
    if any(term in text.lower() for term in ['glass', 'silica', 'oxide', 'temperature']):
        score += 0.3
    
    metrics['quality_score'] = score
    metrics['is_valid'] = score >= 0.5
    
    return metrics


# Example usage
if __name__ == "__main__":
    sample_text = """
    Glass is an amorphous solid material. The glass transition temperature (Tg) 
    is a critical property. Common glass compositions include soda-lime glass 
    (SiO2-Na2O-CaO system) and borosilicate glass.
    
    Manufacturing processes include melting, forming, and annealing. Quality 
    control is essential to prevent defects such as bubbles and stones.
    """
    
    print("Text Processing Utilities")
    print("=" * 50)
    
    cleaned = clean_text(sample_text)
    print(f"\nCleaned text length: {len(cleaned)}")
    
    chunks = chunk_text_semantic(cleaned, max_chunk_size=100)
    print(f"Semantic chunks: {len(chunks)}")
    
    metadata = extract_metadata_from_text(cleaned)
    print(f"\nMetadata: {metadata}")
    
    categories = categorize_text(cleaned)
    print(f"\nCategories: {categories}")
    
    quality = validate_text_quality(cleaned)
    print(f"\nQuality metrics: {quality}")
