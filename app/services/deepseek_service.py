from typing import Union
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse
from app.core.exceptions import BizException
from app.core.logger import logger
from app.services.llm_interface import LLMInterface
from app.llm.deepseekLLM import DeepseekLLM
from app.services.llm_service_factory import LLMServiceFactory


@LLMServiceFactory.auto_register("deepseek")
class DeepseekService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> Union[str, StreamingResponse]:
        try:
            stream = kwargs.pop("stream", False)  # 获取 key 的值，同时删除 key，kwargs 里不再有 stream。
            logger.info(f"[DeepseekService] 模型: {model_name}, kwargs: {kwargs}")
            llm = DeepseekLLM(model_name=model_name, **kwargs)
            messages = [HumanMessage(content=prompt)]

            if stream:
                result = llm.stream_generate(messages, **kwargs)
                if not isinstance(result, StreamingResponse):
                    raise BizException(message="DeepseekService 返回流式响应失败")
                return result

            result = llm.invoke(messages, **kwargs)
            if not isinstance(result, AIMessage):
                raise BizException(message="DeepseekService 返回结果格式错误")
            return result.content
        except Exception as e:
            logger.error(msg="[DeepseekService] 调用失败", exc_info=e)
            raise BizException(message=f"Deepseek 调用失败: {str(e)}") from e
