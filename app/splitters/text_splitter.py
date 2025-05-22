import re
import json
import numpy as np
from typing import List, Optional, Any, Dict, Type

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.core.exceptions import BizException
from app.core.logger import logger
from app.llm.tongyiLLM import TongyiAILLM
from langchain.schema import AIMessage

from app.utils.clean_llm_output import clean_llm_output_robust, extract_json_block
from app.utils.document_process import LegalDocumentExtractor
from app.utils.ocr_util import load_jpg_file_list


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
        if model_path:  # 如果提供了模型路径，使用自定义的 SemanticTextSplitter
            self.splitter = SemanticTextSplitter(model_path=model_path,
                                                 similarity_threshold=similarity_threshold or 0.65)

    def get_chunks(self, text: str) -> List[str]:
        """暴露原始切分结果"""
        return self.splitter.split(text)

    def extract_from_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的文本
    ) -> Dict[str, Any]:
        chunks = self.splitter.split(text)
        merged_result: Dict[str, Any] = {}

        # 初始化输出解析器
        base_parser = PydanticOutputParser(pydantic_object=model_cls)

        # 构建 prompt
        # prompt = PromptTemplate(
        #     template="你是一个法律裁判文书助手，擅长提取各种要素信息,没有的可以不提取，我会给你一段文本:"
        #              "你需要从提供的文本中，**准确、完整地提取结构化信息**:\n\n"
        #              "文本：\n{input}\n\n"
        #              "输出符合要求的JSON对象:{format_instructions}。",
        #     input_variables=["input"],
        #     partial_variables={"format_instructions": base_parser.get_format_instructions()},
        # )
        # 构建增强提示词
        extractor = LegalDocumentExtractor()
        prompt = extractor.build_enhanced_prompt(model_cls=model_cls, parser=base_parser)

        # 包装错误自动修复解析器
        parser = RetryWithErrorOutputParser.from_llm(
            parser=base_parser,
            llm=self.llm,
            max_retries=1
        )

        # 查看字段描述
        print(prompt.partial_variables["field_descriptions"])
        # 构建 chain
        chain = prompt | self.llm

        for i, chunk in enumerate(chunks):
            try:
                prompt_value = prompt.format_prompt(input=chunk)

                text = self._data_cleaning(chain, chunk)
                # 用 RetryWithErrorOutputParser 解析并自动修复错误
                parsed = parser.parse_with_prompt(text, prompt_value)

                # 合并结果
                for k, v in parsed.model_dump().items():
                    if k not in merged_result or merged_result[k] in [None, "", 0, [], {}]:
                        merged_result[k] = v
            except Exception as e:
                raise BizException(message=f"[段落 {i}] 提取失败: {e}")

        return merged_result

    def _data_cleaning(self, chain, chunk):
        response = chain.invoke({"input": chunk})
        # 统一取文本内容
        if isinstance(response, str):
            text_content = response
        elif hasattr(response, "content"):
            text_content = response.content
        else:
            # 兜底，转换成字符串
            text_content = str(response)
        if "<think>" in text_content:
            text_content = clean_llm_output_robust(text_content)
        return extract_json_block(text_content)

    @staticmethod
    def safe_model_validate(model_cls: Type[BaseModel], data: Any) -> BaseModel:
        """安全地将字符串或字典转换为模型实例"""
        if isinstance(data, str):
            try:
                data_dict = json.loads(data)  # 先转字典
                return model_cls.model_validate(data_dict)  # 再转模型实例
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e}\n内容: {data}")
        if isinstance(data, model_cls):
            return data
        elif isinstance(data, dict):
            return model_cls.model_validate(data, strict=False)
        else:
            raise TypeError(f"无法识别的数据类型: {type(data)}")

    def extract_whole_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的完整文本
    ) -> Dict[str, Any]:
        # 设置解析器
        base_parser = PydanticOutputParser(pydantic_object=model_cls)
        # 构建增强提示词
        extractor = LegalDocumentExtractor()
        prompt = extractor.build_enhanced_prompt(model_cls=model_cls, parser=base_parser)
        # 包装错误自动修复解析器
        parser = RetryWithErrorOutputParser.from_llm(
            parser=base_parser,
            llm=self.llm,
            max_retries=1
        )
        # 构建 chain
        chain = prompt | self.llm
        try:
            prompt_value = prompt.format_prompt(input=text)
            resp_text = self._data_cleaning(chain, text)
            # 用 RetryWithErrorOutputParser 解析并自动修复错误
            parsed = parser.parse_with_prompt(resp_text, prompt_value)
            return parsed.model_dump()
        except Exception as e:
            raise BizException(message=f"整段文本结构化抽取失败: {e}")

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
                logger.warning(f"[警告] 关键词精炼失败：{e}")
        return keywords
