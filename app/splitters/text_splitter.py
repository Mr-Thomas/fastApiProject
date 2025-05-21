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

                text = extract_json_block(text_content)
                # 使用安全的模型验证器解析 JSON 或字典
                # parsed = self.safe_model_validate(model_cls, text)
                # 用 RetryWithErrorOutputParser 解析并自动修复错误
                parsed = parser.parse_with_prompt(text, prompt_value)

                # 合并结果
                for k, v in parsed.model_dump().items():
                    if k not in merged_result or merged_result[k] in [None, "", 0, [], {}]:
                        merged_result[k] = v
            except Exception as e:
                raise BizException(message=f"[段落 {i}] 提取失败: {e}")

        return merged_result

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

    def extract_from_text_by_model_bak(
            self,
            model_cls: Type[BaseModel],
            text: str,
    ) -> Dict[str, Any]:

        chunks = self.splitter.split(text)
        merged_result: Dict[str, Any] = {}
        debug_info: List[str] = []

        # 将目标模型转换为 JSON schema 字段说明
        fields_description = ", ".join(model_cls.model_fields.keys())

        prompt = PromptTemplate(
            template=(
                "请从下列文本中提取以下字段的结构化信息：{fields}。\n\n"
                "文本：\n{input}\n\n"
                "请以JSON对象格式输出，禁止输出数组或其他结构。"
            ),
            input_variables=["input"],
            partial_variables={"fields": fields_description},
        )

        # 使用更宽容的 JSON 解析器
        parser = JsonOutputKeyToolsParser(key_name="")  # 输出完整 JSON，无需从某 key 中取值
        chain = prompt | self.llm | parser

        for i, chunk in enumerate(chunks):
            try:
                result: dict = chain.invoke({"input": chunk})
                for k, v in result.items():
                    if v in [None, "", 0]:
                        continue
                    if isinstance(v, list):
                        merged_result.setdefault(k, [])
                        merged_result[k].extend([item for item in v if item not in merged_result[k]])
                    elif isinstance(v, dict):
                        merged_result.setdefault(k, {})
                        merged_result[k].update({kk: vv for kk, vv in v.items() if vv})
                    else:
                        if k not in merged_result or merged_result[k] in [None, "", 0]:
                            merged_result[k] = v
            except Exception as e:
                debug_info.append(f"[段落 {i}] 提取失败: {e}\n内容: {chunk[:100]}...")

        if debug_info:
            raise BizException(
                message="部分段落提取失败，请检查模型结构或输出格式",
                data={"errors": debug_info}
            )

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
