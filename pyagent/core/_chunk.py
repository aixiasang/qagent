from abc import ABC, abstractmethod
import re
from typing import List, Union, Callable, Optional, Any
from bisect import bisect_left
from itertools import accumulate
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from datetime import datetime


@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int
    token_count: int
    tokens: Optional[List[int]] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


@dataclass
class Sentence:
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentence_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.content)


class BaseTokenizer(ABC):

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def count(self, text: str) -> int:
        pass

    def encodes(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decodes(self, token_sequences: List[List[int]]) -> List[str]:
        return [self.decode(tokens) for tokens in token_sequences]

    def counts(self, texts: List[str]) -> List[int]:
        return [self.count(text) for text in texts]


class CharacterTokenizer(BaseTokenizer):

    def __init__(self):
        self.vocab: List[str] = []
        self.token2id: Dict[str, int] = {}

    def _get_or_create_id(self, token: str) -> int:
        if token not in self.token2id:
            token_id = len(self.vocab)
            self.token2id[token] = token_id
            self.vocab.append(token)
            return token_id
        return self.token2id[token]

    def encode(self, text: str) -> List[int]:
        return [self._get_or_create_id(char) for char in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.vocab[tid] for tid in tokens)

    def count(self, text: str) -> int:
        return len(text)

    def __repr__(self) -> str:
        return f"CharacterTokenizer(vocab={len(self.vocab)})"


class WordTokenizer(BaseTokenizer):

    def __init__(self):
        self.vocab: List[str] = []
        self.token2id: Dict[str, int] = {}

    def _get_or_create_id(self, token: str) -> int:
        if token not in self.token2id:
            token_id = len(self.vocab)
            self.token2id[token] = token_id
            self.vocab.append(token)
            return token_id
        return self.token2id[token]

    def encode(self, text: str) -> List[int]:
        return [self._get_or_create_id(word) for word in text.split(" ")]

    def decode(self, tokens: List[int]) -> str:
        return " ".join(self.vocab[tid] for tid in tokens)

    def count(self, text: str) -> int:
        return len(text.split(" "))

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab={len(self.vocab)})"


class ChineseTokenizer(BaseTokenizer):

    def __init__(self):
        try:
            import jieba

            self.jieba = jieba
        except ImportError:
            raise ImportError("jieba required: pip install jieba")

        self.vocab: List[str] = []
        self.token2id: Dict[str, int] = {}

    def _get_or_create_id(self, token: str) -> int:
        if token not in self.token2id:
            token_id = len(self.vocab)
            self.token2id[token] = token_id
            self.vocab.append(token)
            return token_id
        return self.token2id[token]

    def encode(self, text: str) -> List[int]:
        return [self._get_or_create_id(word) for word in self.jieba.cut(text)]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.vocab[tid] for tid in tokens)

    def count(self, text: str) -> int:
        return len(list(self.jieba.cut(text)))

    def __repr__(self) -> str:
        return f"ChineseTokenizer(vocab={len(self.vocab)})"


class RegexTokenizer(BaseTokenizer):

    def __init__(self, pattern: str = r"\w+|[^\w\s]"):
        self.pattern = re.compile(pattern)
        self.vocab: List[str] = []
        self.token2id: Dict[str, int] = {}

    def _get_or_create_id(self, token: str) -> int:
        if token not in self.token2id:
            token_id = len(self.vocab)
            self.token2id[token] = token_id
            self.vocab.append(token)
            return token_id
        return self.token2id[token]

    def encode(self, text: str) -> List[int]:
        return [self._get_or_create_id(token) for token in self.pattern.findall(text)]

    def decode(self, tokens: List[int]) -> str:
        return " ".join(self.vocab[tid] for tid in tokens)

    def count(self, text: str) -> int:
        return len(self.pattern.findall(text))

    def __repr__(self) -> str:
        return f"RegexTokenizer(vocab={len(self.vocab)})"


class CustomTokenizer(BaseTokenizer):

    def __init__(self, counter_fn: Callable[[str], int]):
        self.counter_fn = counter_fn
        self.fn_name = getattr(counter_fn, "__name__", "lambda")

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("CustomTokenizer: encode not supported")

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("CustomTokenizer: decode not supported")

    def count(self, text: str) -> int:
        return self.counter_fn(text)

    def __repr__(self) -> str:
        return f"CustomTokenizer(fn={self.fn_name})"


class TiktokenTokenizer(BaseTokenizer):

    def __init__(self, model: str = "gpt-4"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken required: pip install tiktoken")

        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def __repr__(self) -> str:
        return f"TiktokenTokenizer(model={self.model})"


class TransformersTokenizer(BaseTokenizer):

    def __init__(self, model: str = "bert-base-uncased"):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encodes(self, texts: List[str]) -> List[List[int]]:
        result = self.tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=False
        )
        return result["input_ids"]

    def decodes(self, token_sequences: List[List[int]]) -> List[str]:
        return self.tokenizer.batch_decode(token_sequences, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f"TransformersTokenizer(model={self.model})"


class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self, name: str = "bert-base-uncased"):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("tokenizers required: pip install tokenizers")

        self.tokenizer = Tokenizer.from_pretrained(name)
        self.name = name

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encodes(self, texts: List[str]) -> List[List[int]]:
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        return [enc.ids for enc in encodings]

    def decodes(self, token_sequences: List[List[int]]) -> List[str]:
        return self.tokenizer.decode_batch(token_sequences, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f"HuggingFaceTokenizer(name={self.name})"


class SentencePieceTokenizer(BaseTokenizer):

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece required: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    def encode(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens: List[int]) -> str:
        return self.sp.DecodeIds(tokens)

    def count(self, text: str) -> int:
        return len(self.sp.EncodeAsIds(text))

    def __repr__(self) -> str:
        return f"SentencePieceTokenizer(path={self.model_path})"


class BaseChunker(ABC):

    def __init__(self, tokenizer: Union[BaseTokenizer, Callable] = None):
        if tokenizer is None:
            self.tokenizer = CharacterTokenizer()
        elif isinstance(tokenizer, BaseTokenizer):
            self.tokenizer = tokenizer
        elif callable(tokenizer):
            self.tokenizer = CustomTokenizer(tokenizer)
        else:
            raise TypeError(f"Invalid tokenizer type: {type(tokenizer)}")

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        pass

    def chunk_batch(self, texts: List[str]) -> List[List[Chunk]]:
        return [self.chunk(text) for text in texts]

    def __call__(
        self, text: Union[str, List[str]]
    ) -> Union[List[Chunk], List[List[Chunk]]]:
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list):
            return self.chunk_batch(text)
        else:
            raise ValueError("Input must be a string or list of strings")


class FixedSizeChunker(BaseChunker):

    def __init__(
        self,
        tokenizer: Union[BaseTokenizer, Callable] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ):
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_token_groups(self, tokens: List[int]) -> List[List[int]]:
        groups = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            groups.append(tokens[start:end])
            if end == len(tokens):
                break
            start += step

        return groups

    def _calculate_overlap_lengths(self, token_groups: List[List[int]]) -> List[int]:
        if self.chunk_overlap == 0:
            return [0] * len(token_groups)

        overlap_token_groups = []
        for group in token_groups:
            if len(group) > self.chunk_overlap:
                overlap_token_groups.append(group[-self.chunk_overlap :])
            else:
                overlap_token_groups.append(group)

        overlap_texts = self.tokenizer.decodes(overlap_token_groups)
        return [len(text) for text in overlap_texts]

    def _create_chunks(
        self,
        chunk_texts: List[str],
        token_groups: List[List[int]],
        overlap_lengths: List[int],
    ) -> List[Chunk]:
        chunks = []
        current_char_index = 0
        current_token_index = 0

        for text, tokens, overlap_len in zip(
            chunk_texts, token_groups, overlap_lengths
        ):
            text_stripped = text.strip()

            if not text_stripped:
                continue

            char_start = current_char_index
            char_end = char_start + len(text_stripped)
            token_start = current_token_index
            token_end = token_start + len(tokens)

            chunk = Chunk(
                text=text_stripped,
                start_index=char_start,
                end_index=char_end,
                token_count=self.tokenizer.count(text_stripped),
                tokens=tokens,
                token_start=token_start,
                token_end=token_end,
            )
            chunks.append(chunk)

            current_char_index = char_end - overlap_len
            current_token_index = token_end - self.chunk_overlap

        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        token_groups = self._generate_token_groups(tokens)
        chunk_texts = self.tokenizer.decodes(token_groups)
        overlap_lengths = self._calculate_overlap_lengths(token_groups)

        return self._create_chunks(chunk_texts, token_groups, overlap_lengths)

    def __repr__(self) -> str:
        return f"FixedSizeChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"


class SentenceChunker(BaseChunker):

    def __init__(
        self,
        tokenizer: Union[BaseTokenizer, Callable] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delimiters: List[str] = None,
        include_delimiter: Optional[str] = "prev",
    ):
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be at least 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delimiters = delimiters or [". ", "! ", "? ", "\n"]
        self.include_delimiter = include_delimiter
        self.separator = "✄"

    def _split_sentences(self, text: str) -> List[str]:
        t = text
        for delimiter in self.delimiters:
            if self.include_delimiter == "prev":
                t = t.replace(delimiter, delimiter + self.separator)
            elif self.include_delimiter == "next":
                t = t.replace(delimiter, self.separator + delimiter)
            else:
                t = t.replace(delimiter, self.separator)

        splits = [s for s in t.split(self.separator) if s != ""]

        if not splits:
            return []

        current = ""
        sentences = []
        for s in splits:
            if len(s) < self.min_characters_per_sentence:
                current += s
            elif current:
                current += s
                sentences.append(current)
                current = ""
            else:
                sentences.append(s)

            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""

        if current:
            sentences.append(current)

        return sentences

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        sentence_texts = self._split_sentences(text)
        if not sentence_texts:
            return []

        positions = []
        current_pos = 0
        for sent in sentence_texts:
            positions.append(current_pos)
            current_pos += len(sent)

        token_counts = self.tokenizer.counts(sentence_texts)

        return [
            Sentence(
                text=sent, start_index=pos, end_index=pos + len(sent), token_count=count
            )
            for sent, pos, count in zip(sentence_texts, positions, token_counts)
        ]

    def _create_chunk_from_sentences(self, sentences: List[Sentence]) -> Chunk:
        chunk_text = "".join([s.text for s in sentences]).strip()
        token_count = self.tokenizer.count(chunk_text)

        return Chunk(
            text=chunk_text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
        )

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._prepare_sentences(text)
        if not sentences:
            return []

        token_sums = list(
            accumulate(
                [s.token_count for s in sentences], lambda a, b: a + b, initial=0
            )
        )

        chunks = []
        pos = 0

        while pos < len(sentences):
            target_tokens = token_sums[pos] + self.chunk_size
            split_idx = bisect_left(token_sums, target_tokens) - 1
            split_idx = max(min(split_idx, len(sentences)), pos + 1)

            if split_idx - pos < self.min_sentences_per_chunk:
                if pos + self.min_sentences_per_chunk <= len(sentences):
                    split_idx = pos + self.min_sentences_per_chunk
                else:
                    split_idx = len(sentences)

            chunk_sentences = sentences[pos:split_idx]
            chunks.append(self._create_chunk_from_sentences(chunk_sentences))

            if self.chunk_overlap > 0 and split_idx < len(sentences):
                overlap_tokens = 0
                overlap_idx = split_idx - 1

                while overlap_idx > pos and overlap_tokens < self.chunk_overlap:
                    sent = sentences[overlap_idx]
                    next_tokens = overlap_tokens + sent.token_count
                    if next_tokens > self.chunk_overlap:
                        break
                    overlap_tokens = next_tokens
                    overlap_idx -= 1

                pos = overlap_idx + 1
            else:
                pos = split_idx

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size})"
        )


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        tokenizer: Union[BaseTokenizer, Callable] = None,
        chunk_size: int = 512,
        min_chunk_size: int = 24,
        separators: List[str] = None,
    ):
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")

        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separator: str) -> List[str]:
        if separator == "":
            return list(text)

        sep_marker = "✂"
        temp = text.replace(separator, separator + sep_marker)
        splits = [s for s in temp.split(sep_marker) if s != ""]
        return splits

    def _merge_small_splits(self, splits: List[str], separator: str) -> List[str]:
        if not splits:
            return []

        merged = []
        current = ""

        for split in splits:
            if len(split) < self.min_chunk_size:
                current += split
            elif current:
                current += split
                merged.append(current)
                current = ""
            else:
                merged.append(split)

            if len(current) >= self.min_chunk_size:
                merged.append(current)
                current = ""

        if current:
            if merged and len(current) < self.min_chunk_size:
                merged[-1] += current
            else:
                merged.append(current)

        return merged

    def _recursive_split(
        self, text: str, separators: List[str], start_offset: int
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        text_stripped = text.strip()
        token_count = self.tokenizer.count(text_stripped)

        if token_count <= self.chunk_size:
            return [
                Chunk(
                    text=text_stripped,
                    start_index=start_offset,
                    end_index=start_offset + len(text_stripped),
                    token_count=token_count,
                )
            ]

        if not separators:
            tokens = self.tokenizer.encode(text_stripped)
            token_groups = [
                tokens[i : i + self.chunk_size]
                for i in range(0, len(tokens), self.chunk_size)
            ]
            chunk_texts = self.tokenizer.decodes(token_groups)

            chunks = []
            current_offset = start_offset
            for chunk_text, token_group in zip(chunk_texts, token_groups):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_index=current_offset,
                            end_index=current_offset + len(chunk_text),
                            token_count=len(token_group),
                        )
                    )
                    current_offset += len(chunk_text)

            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = self._split_text(text, separator)
        splits = self._merge_small_splits(splits, separator)

        chunks = []
        current_offset = start_offset

        for split in splits:
            split_stripped = split.strip()
            if not split_stripped:
                continue

            split_token_count = self.tokenizer.count(split_stripped)

            if split_token_count > self.chunk_size:
                sub_chunks = self._recursive_split(
                    split_stripped, remaining_separators, current_offset
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Chunk(
                        text=split_stripped,
                        start_index=current_offset,
                        end_index=current_offset + len(split_stripped),
                        token_count=split_token_count,
                    )
                )

            current_offset += len(split_stripped)

        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        return self._recursive_split(text, self.separators, 0)

    def __repr__(self) -> str:
        return f"RecursiveChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size})"


def _default_embedder(text: str) -> np.ndarray:
    from collections import Counter

    words = text.lower().split()
    if not words:
        return np.zeros(128)
    word_freq = Counter(words)
    vocab = sorted(set(words))
    vector = []
    for word in vocab[:128]:
        vector.append(word_freq[word])
    while len(vector) < 128:
        vector.append(0)
    vector = vector[:128]
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


class SemanticChunker(BaseChunker):

    def __init__(
        self,
        embedder: Optional[Callable[[str], Any]] = None,
        tokenizer: Union[BaseTokenizer, Callable] = None,
        chunk_size: int = 512,
        similarity_threshold: float = 0.5,
        window_size: int = 1,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 10,
        delimiters: List[str] = None,
        include_delimiter: str = "prev",
    ):
        super().__init__(tokenizer)

        if embedder is None:
            self.embedder = _default_embedder
        elif callable(embedder):
            self.embedder = embedder
        else:
            raise TypeError("embedder must be callable or None")

        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delimiters = delimiters or ["。", "！", "？", ". ", "! ", "? ", "\n"]
        self.include_delimiter = include_delimiter
        self.sep = "✄"

    def _embed(self, text: str) -> np.ndarray:
        result = self.embedder(text)
        if isinstance(result, list):
            return np.array(result, dtype=np.float32)
        elif isinstance(result, np.ndarray):
            return result.astype(np.float32)
        else:
            raise ValueError(
                f"embedder must return list or ndarray, got {type(result)}"
            )

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self._embed(text) for text in texts]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _split_sentences(self, text: str) -> List[Sentence]:
        t = text
        for delimiter in self.delimiters:
            if self.include_delimiter == "prev":
                t = t.replace(delimiter, delimiter + self.sep)
            elif self.include_delimiter == "next":
                t = t.replace(delimiter, self.sep + delimiter)
            else:
                t = t.replace(delimiter, self.sep)

        splits = [s for s in t.split(self.sep) if s != ""]

        if not splits:
            token_count = self.tokenizer.count(text)
            return [
                Sentence(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=token_count,
                    sentence_id=0,
                )
            ]

        current = ""
        sentence_texts = []
        for s in splits:
            if len(s) < self.min_characters_per_sentence:
                current += s
            elif current:
                current += s
                sentence_texts.append(current)
                current = ""
            else:
                sentence_texts.append(s)

            if len(current) >= self.min_characters_per_sentence:
                sentence_texts.append(current)
                current = ""

        if current:
            sentence_texts.append(current)

        if not sentence_texts:
            token_count = self.tokenizer.count(text)
            return [
                Sentence(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=token_count,
                    sentence_id=0,
                )
            ]

        token_counts = self.tokenizer.counts(sentence_texts)

        sentences = []
        current_pos = 0
        for i, (sent_text, token_count) in enumerate(zip(sentence_texts, token_counts)):
            sent = Sentence(
                text=sent_text,
                start_index=current_pos,
                end_index=current_pos + len(sent_text),
                token_count=token_count,
                sentence_id=i,
            )
            sentences.append(sent)
            current_pos += len(sent_text)

        return sentences

    def _calculate_similarities(self, sentences: List[Sentence]) -> List[float]:
        if len(sentences) <= self.window_size:
            return []

        sentence_texts = [s.text for s in sentences]
        embeddings = self._embed_batch(sentence_texts)
        similarities = []

        for i in range(len(sentences) - self.window_size):
            if self.window_size == 1:
                window_emb = embeddings[i]
            else:
                window_embs = embeddings[i : i + self.window_size]
                window_emb = np.mean(window_embs, axis=0)

            next_emb = embeddings[i + self.window_size]
            sim = self._cosine_similarity(window_emb, next_emb)
            similarities.append(sim)

        return similarities

    def _find_split_points(self, similarities: List[float]) -> List[int]:
        if not similarities:
            return []

        split_points = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_idx = i + self.window_size
                if (
                    not split_points
                    or split_idx - split_points[-1] >= self.min_sentences_per_chunk
                ):
                    split_points.append(split_idx)

        return split_points

    def _group_sentences(
        self, sentences: List[Sentence], split_points: List[int]
    ) -> List[List[Sentence]]:
        if not split_points:
            return [sentences]

        groups = []
        prev = 0
        for point in split_points:
            if point > prev:
                groups.append(sentences[prev:point])
                prev = point

        if prev < len(sentences):
            groups.append(sentences[prev:])

        return groups

    def _create_chunks_from_groups(
        self, sentence_groups: List[List[Sentence]]
    ) -> List[Chunk]:
        chunks = []
        current_index = 0

        for group in sentence_groups:
            group_text = "".join(s.text for s in group).strip()

            if not group_text:
                continue

            group_token_count = self.tokenizer.count(group_text)

            if group_token_count > self.chunk_size:
                for sent in group:
                    sent_text = sent.text.strip()
                    if sent_text:
                        chunks.append(
                            Chunk(
                                text=sent_text,
                                start_index=current_index,
                                end_index=current_index + len(sent_text),
                                token_count=self.tokenizer.count(sent_text),
                            )
                        )
                        current_index += len(sent_text)
            else:
                chunks.append(
                    Chunk(
                        text=group_text,
                        start_index=current_index,
                        end_index=current_index + len(group_text),
                        token_count=group_token_count,
                    )
                )
                current_index += len(group_text)

        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)

        if len(sentences) <= self.min_sentences_per_chunk:
            token_count = self.tokenizer.count(text)
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=token_count,
                )
            ]

        similarities = self._calculate_similarities(sentences)
        split_points = self._find_split_points(similarities)
        sentence_groups = self._group_sentences(sentences, split_points)
        chunks = self._create_chunks_from_groups(sentence_groups)

        return chunks

    def __repr__(self) -> str:
        return f"SemanticChunker(threshold={self.similarity_threshold}, window={self.window_size})"


if __name__ == "__main__":
    print("=" * 80)
    print("Chunker test")
    print("=" * 80)

    with open("test_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentence_chunker = SemanticChunker()
    chunks = sentence_chunker.chunk(text)

    print(f"Total chunks: {len(chunks)}")
    print(f"Original text length: {len(text)} chars\n")

    # 验证边界空白移除
    all_clean = True
    has_whitespace_count = 0

    for i, chunk in enumerate(chunks[:20]):  # 只检查前20个
        has_whitespace = chunk.text != chunk.text.strip()
        if has_whitespace:
            all_clean = False
            has_whitespace_count += 1

        status = "❌" if has_whitespace else "✅"
        print(f"{status} Chunk {i}: {chunk.token_count} tokens - {chunk.text[:60]}")

    print(f"\n... 还有 {len(chunks) - 20} 个chunks")

    print("\n" + "=" * 80)
    if all_clean:
        print("✅ 所有检查的 chunks 边界都是干净的（无前后空白）！")
    else:
        print(f"❌ 发现 {has_whitespace_count} 个 chunks 有边界空白")
    print("=" * 80)
