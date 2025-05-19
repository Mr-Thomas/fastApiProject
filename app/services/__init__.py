from .llm_factory import LLMServiceFactory
from .ollama_service import OllamaService
from .tongyi_service import TongyiAiService
from .zhipuai_service import ZhipuAiService

# 注册 Ollama 服务
LLMServiceFactory.register("ollama", OllamaService)
# zhipuai
LLMServiceFactory.register("zhipuai", ZhipuAiService)
# tongyi
LLMServiceFactory.register("tongyi", TongyiAiService)
