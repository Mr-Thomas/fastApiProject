from langchain_core.messages import HumanMessage
from app.core.exceptions import BizException
from app.services.llm_interface import LLMInterface
from app.llm.ollamaChatLLM import OllamaChatLLM


class OllamaService(LLMInterface):

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            llm = OllamaChatLLM(model=model_name)
            messages = [HumanMessage(content=prompt)]
            result = llm.invoke(messages)
            return result.content
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"ollamaLLM 调用失败: {str(e)}")
