from langchain_ollama import OllamaLLM

from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_interface import LLMInterface
from app.utils.clean_llm_output import clean_llm_output


class OllamaService(LLMInterface):
    def __init__(self):
        base_url = settings.ollama_url
        if not base_url:
            raise BizException(message="OLLAMA_URL 未配置")
        # 将 base_url 保存为实例属性
        self.base_url = base_url

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            llm = OllamaLLM(base_url=self.base_url, model=model_name)
            result = llm.invoke(prompt)
            return clean_llm_output("think", result)
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"ollamaLLM 调用失败: {str(e)}")
