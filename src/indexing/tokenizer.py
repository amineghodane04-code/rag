import re
from typing import List, Set

# Common stop words to avoid noise in BM25 scoring
STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "of",
    "to",
    "for",
    "with",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
}


def split_identifier(identifier: str) -> List[str]:
    """
    Split snake_case, PascalCase, camelCase, or SCREAMING_SNAKE_CASE identifiers
    into their constituent sub-words while also retaining the full identifier.

    Examples:
      - 'tie_word_embeddings' -> ['tie_word_embeddings', 'tie', 'word', 'embeddings']
      - 'HybridKVCacheCoordinator' -> ['hybridkvcachecoordinator', 'hybrid', 'kv',
                                       'cache', 'coordinator']
      - 'VLLM_RPC_TIMEOUT' -> ['vllm_rpc_timeout', 'vllm', 'rpc', 'timeout']
      - 'Gemma2DecoderLayer' -> ['gemma2decoderlayer', 'gemma', '2', 'decoder', 'layer']
    """
    tokens = [identifier.lower()]

    # Split camelCase / PascalCase while preserving numbers
    # e.g., 'Gemma2DecoderLayer' -> 'Gemma_2_Decoder_Layer'
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    # Split alphanumeric transitions (e.g., 'gemma2' -> 'gemma_2')
    s3 = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s2)
    s4 = re.sub(r"([0-9])([a-zA-Z])", r"\1_\2", s3)

    parts = [p.lower() for p in s4.split("_") if p]
    for p in parts:
        if len(p) > 1 and p not in tokens:
            tokens.append(p)

    return list(dict.fromkeys(tokens))


def tokenize_for_bm25(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenizes text for lexical search (BM25) by preserving exact identifiers
    and breaking them down into searchable sub-tokens.
    """
    # Extract code identifiers and words
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    tokens: List[str] = []

    for raw in raw_tokens:
        sub_tokens = split_identifier(raw)
        for tok in sub_tokens:
            if remove_stopwords and tok in STOP_WORDS:
                continue
            tokens.append(tok)

    return tokens
