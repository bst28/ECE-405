from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import regex as re


# This is the GPT-2 style regex used to split text into "pieces"
# (words, numbers, punctuation, spaces, etc.)
GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _byte_symbols(token: str) -> Tuple[bytes, ...]:
    """
    Turn one text piece into a tuple of single-byte symbols.
    Example: "hi" -> (b'h', b'i')  (but in UTF-8 bytes)
    """
    b = token.encode("utf-8")
    return tuple(bytes([x]) for x in b)


def _get_pair_counts(word_freq: Counter[Tuple[bytes, ...]]) -> Counter[Tuple[bytes, bytes]]:
    """
    Count how often each adjacent byte-pair appears across all words/pieces.
    """
    pair_counts: Counter[Tuple[bytes, bytes]] = Counter()

    # symbols is a tuple of bytes like (b'a', b'b', b'c')
    # freq is how many times that sequence appears
    for symbols, freq in word_freq.items():
        if len(symbols) < 2:
            continue

        # Count every adjacent pair inside this sequence
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq

    return pair_counts


def _merge_word(symbols: Tuple[bytes, ...], pair: Tuple[bytes, bytes], merged: bytes) -> Tuple[bytes, ...]:
    """
    Replace all occurrences of `pair` in one word/piece with the merged byte token.
    """
    a, b = pair
    out: List[bytes] = []
    i = 0
    n = len(symbols)

    # Walk left-to-right and merge whenever we see (a followed by b)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1

    return tuple(out)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
      vocab: dict[int, bytes]  (token_id -> token_bytes)
      merges: list[tuple[bytes, bytes]]  (ordered by creation)
    """
    # Basic input check
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    # 1) Read training text as normal UTF-8 text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Split text into GPT-2 style pieces (words/punctuation/spaces)
    pieces = GPT2_PATTERN.findall(text)

    # Count each piece as a tuple of single-byte symbols
    word_freq: Counter[Tuple[bytes, ...]] = Counter()
    for p in pieces:
        syms = _byte_symbols(p)
        if syms:
            word_freq[syms] += 1

    # 3) Start vocab with the 256 single-byte tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens after the byte vocab
    # (They do not affect merges because they aren't in word_freq)
    next_id = 256
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    # Store merge rules in the order we learn them
    merges: List[Tuple[bytes, bytes]] = []

    # Compute how many merges we are allowed to do to hit vocab_size
    # total = 256 bytes + specials + merges
    max_merges = vocab_size - (256 + len(special_tokens))
    if max_merges <= 0:
        return vocab, merges

    # 4) Learn merges one at a time
    for _ in range(max_merges):
        # Count all adjacent byte-pairs across the dataset
        pair_counts = _get_pair_counts(word_freq)
        if not pair_counts:
            break

        # Choose the most frequent pair
        # If there is a tie, choose the lexicographically smallest pair for consistency
        best_freq = max(pair_counts.values())
        best_pairs = [pair for pair, c in pair_counts.items() if c == best_freq]
        best_pair = min(best_pairs)

        # Create the merged byte token
        a, b = best_pair
        merged = a + b

        # Save this merge rule and add merged token to vocab
        merges.append((a, b))
        vocab[next_id] = merged
        next_id += 1

        # Apply the merge to every word/piece and rebuild the counter
        new_word_freq: Counter[Tuple[bytes, ...]] = Counter()
        for symbols, freq in word_freq.items():
            new_symbols = _merge_word(symbols, best_pair, merged)
            new_word_freq[new_symbols] += freq
        word_freq = new_word_freq

    return vocab, merges