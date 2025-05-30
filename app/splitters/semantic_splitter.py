import re
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class SemanticTextSplitter:
    """
    语义切分器 SemanticTextSplitter：
    使用句向量和余弦相似度将文本分块
    """

    def __init__(self,
                 model_path: str = "./local_models/bge-small-zh",  # 本地路径
                 similarity_threshold: float = 0.65):
        self.model = SentenceTransformer(model_path)
        self.threshold = similarity_threshold

    def split(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                chunks.append("".join(current_chunk))
                current_chunk = []
            current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """支持中文标点 + 段落换行的句子切分"""
        # 首先按中文句子标点切分（保留标点）
        punctuated = re.split(r'([。！!？?；;])', text)
        sentences = ["".join(x).strip() for x in zip(punctuated[::2], punctuated[1::2])]

        # 拼接最后一个残句（如果标点数是奇数）
        if len(punctuated) % 2 != 0:
            sentences.append(punctuated[-1].strip())

        # 补充：再按段落换行切割（\n 两次或多个空行作为段落）
        final_sentences = []
        for sent in sentences:
            chunks = re.split(r'\n{1,}|\r\n{1,}', sent)
            final_sentences.extend([chunk.strip() for chunk in chunks if chunk.strip()])

        return final_sentences

    def _cosine_similarity(self, vec1, vec2) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
