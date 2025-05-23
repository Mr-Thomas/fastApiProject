from typing import Union

from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse

from app.core.exceptions import BizException
from app.core.logger import logger
from app.services.llm_interface import LLMInterface
from app.llm.tongyiLLM import TongyiAILLM


class TongyiAiService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> Union[str, StreamingResponse]:
        try:
            stream = kwargs.get("stream", False)
            logger.info(f"[TongyiAiService] 模型: {model_name}, kwargs: {kwargs}")
            llm = TongyiAILLM(model_name=model_name, **kwargs)
            messages = [HumanMessage(content=prompt)]
            if stream:
                result = llm.stream_generate(messages, **kwargs)
                if not isinstance(result, StreamingResponse):
                    raise BizException(message="LLM 返回流式响应失败")
                return result

            result = llm.invoke(messages, **kwargs)
            if not isinstance(result, AIMessage):
                raise BizException(message="LLM 返回结果格式错误")
            return result.content
        except Exception as e:
            logger.error(msg="[TongyiAiService] 调用失败", exc_info=e)
            raise BizException(message=f"tongyiLLM 调用失败: {str(e)}") from e
