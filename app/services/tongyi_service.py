from langchain_core.messages import HumanMessage
from app.core.exceptions import BizException
from app.services.llm_interface import LLMInterface
from app.llm.tongyiLLM import TongyiAILLM


class TongyiAiService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            llm = TongyiAILLM(model_name=model_name, **kwargs)
            messages = [HumanMessage(content=prompt)]
            result = llm.invoke(messages, **kwargs)
            return result.content
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"tongyiLLM 调用失败: {str(e)}") from e
