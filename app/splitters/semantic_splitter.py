import re
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer

from app.core.logger import logger


class SemanticTextSplitter:
    """
    SemanticTextSplitter（无 fallback）版本：
    1. 结构清晰的语义切分器
    2. 支持动态相似度阈值
    3. 支持最大分块数控制
    """

    def __init__(
            self,
            model_path: str = "./local_models/bge-small-zh",
            similarity_threshold: float = 0.65,
            min_chunk_length: int = 1500,
            max_chunk_length: int = 2000,
            dynamic_threshold: bool = True,
            max_chunk_count: int = 100
    ):
        self.model = SentenceTransformer(model_path)
        self.base_threshold = similarity_threshold
        self.min_length = min_chunk_length
        self.max_length = max_chunk_length
        self.dynamic_threshold = dynamic_threshold
        self.max_chunk_count = max_chunk_count

        self.sentence_delimiters = re.compile(r'[^。！？!？；;\n\r]+[。！？!？；;\n\r]*')
        self.paragraph_delimiters = re.compile(r'\n{2,}|\r\n{2,}|[\f\v]+')

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)

        if len(sentences) == 1:
            return [text] if self.min_length <= len(text) <= self.max_length else []

        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            normalize_embeddings=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        chunks = []
        current_chunk = [sentences[0]]
        current_emb = embeddings[0]

        for i in range(1, len(sentences)):
            new_sentence = sentences[i]
            new_emb = embeddings[i]

            current_length = sum(len(s) for s in current_chunk)
            if current_length + len(new_sentence) > self.max_length:
                if current_length >= self.min_length:
                    chunks.append("".join(current_chunk))
                current_chunk = [new_sentence]
                current_emb = new_emb
                continue

            sim = np.dot(current_emb, new_emb)
            threshold = self._get_dynamic_threshold(current_length)

            if sim < threshold and self._should_split(current_chunk, new_sentence):
                if current_length >= self.min_length:
                    chunks.append("".join(current_chunk))
                current_chunk = [new_sentence]
                current_emb = new_emb
            else:
                current_chunk.append(new_sentence)
                current_emb = self._update_embedding(current_emb, new_emb, len(current_chunk))

            if len(chunks) >= self.max_chunk_count:
                logger.warning("Reached max_chunk_count limit.")
                break

        final_chunk = "".join(current_chunk)
        if len(final_chunk) >= self.min_length:
            chunks.append(final_chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        paragraphs = [p for p in self.paragraph_delimiters.split(text) if p.strip()]
        sentences = []

        for para in paragraphs:
            for match in self.sentence_delimiters.finditer(para):
                sentence = match.group().strip()
                if sentence:
                    sentences.append(sentence)

        return sentences

    def _get_dynamic_threshold(self, current_length: int) -> float:
        if not self.dynamic_threshold:
            return self.base_threshold
        length_factor = min(current_length / self.max_length, 1.0)
        return max(self.base_threshold - 0.15 * (length_factor ** 1.5), 0.45)

    def _should_split(self, current_chunk: List[str], next_sentence: str) -> bool:
        current_length = sum(len(s) for s in current_chunk)
        if current_length >= self.max_length * 0.95:
            return True

        if re.match(r'^[#*=_.\-]{3,}', next_sentence.strip()):
            return True

        return False

    def _update_embedding(self, current: np.ndarray, new: np.ndarray, count: int) -> np.ndarray:
        alpha = min(0.8, 1.0 / np.sqrt(count))
        return current * (1 - alpha) + new * alpha

    def batch_split(self, texts: List[str]) -> List[List[str]]:
        results = []
        for text in texts:
            try:
                results.append(self.split(text))
            except Exception as e:
                logger.error(f"Error splitting text: {str(e)}")
                results.append([text])
        return results
