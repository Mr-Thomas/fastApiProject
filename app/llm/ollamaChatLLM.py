from typing import List, Optional, Any, Mapping
from langchain.schema import AIMessage, BaseMessage, ChatResult, ChatGeneration
from langchain.chat_models.base import BaseChatModel
from langchain_ollama import OllamaLLM

from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("ollama")
class OllamaChatLLM(BaseChatModel):
    """基于 OllamaLLM 封装的 ChatModel，兼容 LangChain 聊天接口"""

    def __init__(self, *, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if base_url is None:
            base_url = settings.ollama_url
            if not base_url:
                raise BizException(message="OLLAMA_URL 未配置")
        self._ollama_llm = OllamaLLM(model=model, base_url=base_url, **kwargs)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        # 调用内部 OllamaLLM 的文本生成接口
        llm_result = self._ollama_llm._generate([prompt], stop=stop, **kwargs)
        text = llm_result.generations[0][0].text
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        # 简单拼接消息内容，带角色提示，或者根据需要调整格式
        prompt = ""
        for msg in messages:
            role = msg.type  # human, ai, system
            if role == "human":
                prefix = "User: "
            elif role == "ai":
                prefix = "Assistant: "
            elif role == "system":
                prefix = "System: "
            else:
                prefix = "User: "
            prompt += f"{prefix}{msg.content}\n"
        prompt += "Assistant: "  # 期望模型接着生成回复
        return prompt

    @property
    def _llm_type(self) -> str:
        return "ollama-chat-llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self._ollama_llm.model, **self._ollama_llm.dict()}
