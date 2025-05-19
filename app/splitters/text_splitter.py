import re
import json
import numpy as np
from typing import List, Optional, Any, Dict, Type

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.core.exceptions import BizException
from app.llm.tongyiLLM import TongyiAILLM
from langchain.schema import AIMessage

from app.utils.ocr_util import load_jpg_file, load_jpg_file_list


class SemanticTextSplitter:
    def __init__(self,
                 model_path: str = "./models/bge-small-zh",  # 本地路径
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
        """使用中文语句标点切句"""
        raw = re.split(r'(。|！|\!|？|\?|；|;)', text)
        sentences = ["".join(x).strip() for x in zip(raw[::2], raw[1::2])]
        return [s for s in sentences if s]

    def _cosine_similarity(self, vec1, vec2) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class KeywordExtractor:
    def __init__(self, llm, model_path: Optional[str] = None,
                 similarity_threshold: Optional[float] = None):
        self.llm = llm
        if model_path:
            self.splitter = SemanticTextSplitter(model_path=model_path,
                                                 similarity_threshold=similarity_threshold or 0.65)

    def get_chunks(self, text: str) -> List[str]:
        """暴露原始切分结果"""
        return self.splitter.split(text)

    def extract_from_chunks(
            self,
            chunks: List[str],
            prompt_template: str
    ) -> Dict[str, Any]:
        """支持自定义 prompt 模板从多个 chunks 抽取合并结构信息"""
        merged_result: Dict[str, Any] = {}

        for i, chunk in enumerate(chunks):
            full_prompt = f"{prompt_template}\n\n{chunk}"
            try:
                response = self.llm.invoke(full_prompt)
                content = response.content if isinstance(response, AIMessage) else str(response)

                if not content.strip():
                    continue

                content = re.sub(r'^```(?:json)?|\s*```\s*$', '', content, flags=re.MULTILINE).strip()
                extracted = json.loads(content)

                if isinstance(extracted, dict):
                    for k, v in extracted.items():
                        if k not in merged_result or merged_result[k] in [None, "", 0]:
                            merged_result[k] = v
            except Exception as e:
                raise BizException(message=f"extract_from_chunks失败: {e}")

        return merged_result

    def extract_from_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的文本
    ) -> Dict[str, Any]:
        chunks = self.splitter.split(text)
        merged_result: Dict[str, Any] = {}

        # 设置解析器（自动将输出映射为指定模型）
        parser = PydanticOutputParser(pydantic_object=model_cls)

        prompt = PromptTemplate(
            template="从下列文本中提取结构化信息:\n{input}\n{format_instructions}",
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 构建 chain
        chain = prompt | self.llm | parser

        for i, chunk in enumerate(chunks):
            try:
                instance: BaseModel = chain.invoke({"input": chunk})
                for k, v in instance.model_dump().items():
                    if k not in merged_result or merged_result[k] in [None, "", 0]:
                        merged_result[k] = v
            except Exception as e:
                raise BizException(message=f"[段落 {i}] 提取失败: {e}")

        return merged_result

    def extract_whole_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的完整文本
    ) -> Dict[str, Any]:
        # 设置解析器
        parser = PydanticOutputParser(pydantic_object=model_cls)

        prompt = PromptTemplate(
            template="从下列文本中提取结构化信息:\n{input}\n{format_instructions}",
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 构建 chain
        chain = prompt | self.llm | parser

        try:
            instance: BaseModel = chain.invoke({"input": text})
            return instance.model_dump()
        except Exception as e:
            raise BizException(message=f"整段文本结构化抽取失败: {e}")

    def extract_from_text(self, text: str, prompt: str) -> Dict[str, Any]:
        chunks = self.get_chunks(text)
        return self.extract_from_chunks(chunks, prompt)

    def refine_keywords(self, keywords: List[str], top_k: int = 20) -> List[str]:
        prompt = (
            f"以下是提取的关键词：\n{keywords}\n\n"
            f"请从中筛选出最具代表性的前 {top_k} 个关键词，输出为 JSON 数组："
        )
        response = self.llm.invoke(prompt)
        if isinstance(response, AIMessage):
            try:
                refined = json.loads(response.content)
                if isinstance(refined, list):
                    return refined
            except Exception as e:
                print(f"[警告] 关键词精炼失败：{e}")
        return keywords


if __name__ == '__main__':
    llm = TongyiAILLM(api_key="sk-762be7bbcf42464fbbad8e818b34bec7")
    extractor = KeywordExtractor(llm, model_path="../models/bge-small-zh", similarity_threshold=0.75)
    file_path = ["../documents/page_1.jpg", "../documents/page_2.jpg"]
    text = load_jpg_file_list(file_path)
    text_list = extractor.get_chunks(text=text)
    json_list = extractor.extract_from_text(text=text,
                                            prompt="请从以下文本提取原告(姓名、地址)、被告(姓名、地址)、小区名称、逾期费用、物业费，分别用plaintiff_name,plaintiff_address,defendant_name,defendant_address,cell_name,overdue_charge,property_fee表示，响应格式为json串，不要包含其他任何内容或者格式")
    print(json_list)
