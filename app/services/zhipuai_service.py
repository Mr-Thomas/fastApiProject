from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_interface import LLMInterface
from zhipuai import ZhipuAI

from app.llm.zhipuLLM import ZhipuAILLM


class ZhipuAiService(LLMInterface):

    def __init__(self):
        zhipuai_api_key = settings.zhipuai_api_key
        if not zhipuai_api_key:
            raise BizException(message="zhipuai_api_key 未配置")
        self.client = ZhipuAI(api_key=zhipuai_api_key)  # 初始化 client

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            llm = ZhipuAILLM(client=self.client, model_name=model_name)
            messages = [HumanMessage(content=prompt)]
            result = llm.invoke(messages)
            return result.content
        except Exception as e:
            # 如果你有统一异常处理器，也可以让它抛出 BizException 或记录日志
            raise BizException(message=f"ZhipuAI 调用失败: {str(e)}") from e
