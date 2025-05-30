from typing import Type, Optional, Any, Dict, List
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from app.core.exceptions import BizException
from app.utils.clean_llm_output import clean_llm_output_robust, extract_json_block
from app.utils.document_process import LegalDocumentExtractor
from app.splitters.semantic_splitter import SemanticTextSplitter


class KeywordExtractor:
    """
    关键词结构抽取器 KeywordExtractor：负责调用 LLM 做结构化信息提取，支持逐段和整段两种方式
    """

    def __init__(self, llm: BaseChatModel,
                 model_path: Optional[str] = None,
                 similarity_threshold: Optional[float] = None):
        self.llm = llm
        if model_path:
            self.splitter = SemanticTextSplitter(model_path=model_path,
                                                 similarity_threshold=similarity_threshold or 0.65)

    def get_chunks(self, text: str) -> List[str]:
        if not self.splitter:
            raise BizException(message="SemanticTextSplitter is not initialized")
        return self.splitter.split(text)

    def extract_from_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的文本
    ) -> Dict[str, Any]:
        """
        多段文本中提取指定的结构化数据：
        该方法将文本切分成多个段落，并对每个段落进行模型解析，最后将所有段落的解析结果合并。
        """
        chunks = self.splitter.split(text)
        merged_result: Dict[str, Any] = {}

        # 初始化输出解析器
        base_parser = PydanticOutputParser(pydantic_object=model_cls)

        # 构建增强提示词
        extractor = LegalDocumentExtractor()
        prompt = extractor.build_enhanced_prompt(model_cls=model_cls, parser=base_parser)
        # 查看字段描述
        print(prompt.partial_variables["field_descriptions"])

        # 包装错误自动修复解析器
        parser = RetryWithErrorOutputParser.from_llm(parser=base_parser, llm=self.llm)

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

    def extract_whole_text_by_model(
            self,
            model_cls: Type[BaseModel],  # 结构体模型，如 Person
            text: str,  # 待提取的完整文本
    ) -> Dict[str, Any]:
        """
        整段文本中提取指定的结构化数据：
        该方法将整个文本进行模型解析，最后输出解析结果。
        """
        # 设置解析器
        base_parser = PydanticOutputParser(pydantic_object=model_cls)

        # 构建增强提示词
        extractor = LegalDocumentExtractor()
        prompt = extractor.build_enhanced_prompt(model_cls=model_cls, parser=base_parser)

        # 包装错误自动修复解析器
        parser = RetryWithErrorOutputParser.from_llm(parser=base_parser, llm=self.llm)
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
