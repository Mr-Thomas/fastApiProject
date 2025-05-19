from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.core.exceptions import BizException
from app.llm.tongyiLLM import TongyiAILLM
from app.services.llm_interface import LLMInterface


class TongyiAiService(LLMInterface):

    def __init__(self):
        api_key = settings.dashscope_api_key
        if not api_key:
            raise BizException(message="dashscope_api_key 未配置")
        llm = TongyiAILLM(api_key=api_key)
        self.llm = llm

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        is_stream = kwargs.get('stream', False)  # 从 kwargs 中获取是否流式输出的参数
        try:
            messages = [HumanMessage(content=prompt)]
            if model_name:
                self.llm.model = model_name
            result = self.llm.invoke(messages, stream=is_stream)
            return result.content
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"tongyiLLM 调用失败: {str(e)}") from e
