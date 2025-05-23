from typing import Union
from langchain_core.messages import HumanMessage, AIMessage
from starlette.responses import StreamingResponse
from app.core.exceptions import BizException
from app.core.logger import logger
from app.services.llm_interface import LLMInterface
from app.llm.ollamaChatLLM import OllamaChatLLM


class OllamaService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> Union[str, StreamingResponse]:
        try:
            stream = kwargs.get("stream", False)
            logger.info(f"[OllamaService] 模型: {model_name}, kwargs: {kwargs}")
            llm = OllamaChatLLM(model_name=model_name, **kwargs)
            messages = [HumanMessage(content=prompt)]

            if stream:
                result = llm.stream_generate(messages, **kwargs)
                if not isinstance(result, StreamingResponse):
                    raise BizException(message="OllamaServiceLLM 返回流式响应失败")
                return result

            result = llm.invoke(messages, **kwargs)
            if not isinstance(result, AIMessage):
                raise BizException(message="OllamaServiceLLM 返回结果格式错误")
            return result.content
        except Exception as e:
            logger.error(msg="[OllamaService] 调用失败", exc_info=e)
            raise BizException(message=f"ollamaLLM 调用失败: {str(e)}") from e
