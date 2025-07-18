from langchain_core.messages import HumanMessage, AIMessage
from app.core.exceptions import BizException
from fastapi.responses import StreamingResponse
from typing import Union
from app.services.llm_interface import LLMInterface
from app.llm.zhipuLLM import ZhipuAILLM
from app.services.llm_service_factory import LLMServiceFactory


@LLMServiceFactory.auto_register("zhipuai")  # noqa: F821
class ZhipuAiService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> Union[str, StreamingResponse]:
        try:
            stream = kwargs.get("stream", False)
            llm = ZhipuAILLM(model_name=model_name, **kwargs)
            messages = [HumanMessage(content=prompt)]
            if stream:
                result = llm.stream_generate(messages, **kwargs)
                if not isinstance(result, StreamingResponse):
                    raise BizException(message="ZhipuAiServiceLLM 返回流式响应失败")
                return result

            result = llm.invoke(messages, **kwargs)
            if not isinstance(result, AIMessage):
                raise BizException(message="ZhipuAiServiceLLM 返回结果格式错误")
            return result.content
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"ZhipuAI 调用失败: {str(e)}") from e
