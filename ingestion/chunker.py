"""
Glass Expert AI — Text Chunker
Splits extracted document text into overlapping chunks that respect
sentence and paragraph boundaries for better retrieval quality.
"""

import re
import tiktoken
from loguru import logger


# Use the cl100k_base tokenizer (same as GPT-4 / bge-m3 compatible)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    language: str = "en",
) -> list[str]:
    """
    Split text into overlapping chunks respecting sentence boundaries.

    Args:
        text:       Full document text to chunk.
        chunk_size: Maximum tokens per chunk (default: 512).
        overlap:    Token overlap between consecutive chunks (default: 64).
        language:   'en' or 'fa' — affects sentence splitting.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Split into sentences
    sentences = _split_into_sentences(text, language)
    logger.debug(f"Split into {len(sentences)} sentences")

    # Build chunks from sentences
    chunks = _build_chunks(sentences, chunk_size, overlap)
    logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    return chunks


def _split_into_sentences(text: str, language: str) -> list[str]:
    """Split text into sentences using regex, handling both English and Farsi."""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if language == "fa":
        # Farsi sentence delimiters: Persian period (۔), exclamation (!)
        # and question mark (؟) in addition to standard ones
        pattern = r"(?<=[.!?؟۔])\s+"
    else:
        # English sentence splitting
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"

    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def _build_chunks(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Build overlapping chunks from a list of sentences."""
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(_ENCODER.encode(sentence))

        # If a single sentence exceeds chunk_size, split it by words
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0
            # Force-split the long sentence
            word_chunks = _split_long_sentence(sentence, chunk_size)
            chunks.extend(word_chunks)
            continue

        # If adding this sentence would exceed chunk_size, save current chunk
        if current_token_count + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Create overlap: keep last N sentences that fit within overlap tokens
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = len(_ENCODER.encode(s))
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break

            current_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Add the final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Force-split a sentence that exceeds chunk_size by words."""
    words = sentence.split()
    chunks = []
    current_words = []
    current_tokens = 0

    for word in words:
        word_tokens = len(_ENCODER.encode(word))
        if current_tokens + word_tokens > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
            current_tokens = 0
        current_words.append(word)
        current_tokens += word_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_ENCODER.encode(text))
