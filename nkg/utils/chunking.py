from functools import lru_cache

from transformers import AutoTokenizer

DEFAULT_CHUNK_TOKENIZER = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


@lru_cache(maxsize=None)
def _get_tokenizer(model: str):
    """Load each tokenizer once per process (construction calls the chunker once per chunk, from many threads)."""
    return AutoTokenizer.from_pretrained(model)


def chunk_text_by_tokens(model: str = None, text: str = None, chunk_size: int = 600, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks of chunk_size tokens (stride chunk_size - overlap)."""
    tokenizer = _get_tokenizer(model or DEFAULT_CHUNK_TOKENIZER)
    tokens = tokenizer.encode(text)
    chunks = []
    step = max(1, chunk_size - overlap)  # at least 1, so the loop always advances
    for i in range(0, len(tokens), step):
        chunks.append(tokenizer.decode(tokens[i:i + chunk_size]))
    return chunks
