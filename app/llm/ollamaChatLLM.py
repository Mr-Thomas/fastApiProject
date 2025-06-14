import json
from typing import List, Optional, Any, Mapping, Iterator
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import Field
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("ollama")
class OllamaChatLLM(BaseChatModel):
    """基于 OllamaLLM 封装的 ChatModel，兼容 LangChain 聊天接口"""
    model_name: str = Field(default="qwen2.5:14b", description="模型名称")
    temperature: float = Field(default=0.8, description="温度参数")
    system_prompt: Optional[str] = Field(default=None, description="系统级指令")

    def __init__(self, *, model_name: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = kwargs.pop("temperature", 0.8)
        self.system_prompt = kwargs.pop("system_prompt", None)
        if base_url is None:
            base_url = settings.ollama_url
            if not base_url:
                raise BizException(message="OLLAMA_URL 未配置")

        self._chat_ollama = ChatOllama(model=self.model_name,
                                       base_url=base_url,
                                       temperature=self.temperature,
                                       **kwargs)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        # 统一处理消息格式
        processed_messages = self._process_messages(messages)

        # 删除 temperature 防止重复传入（初始化已经传入了）
        kwargs.pop("temperature", None)

        llm_result = self._chat_ollama._generate(messages=processed_messages,
                                                 stop=stop,
                                                 **kwargs)
        return llm_result

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """原生流式生成方法"""
        processed_messages = self._process_messages(messages)
        yield from self._chat_ollama._stream(
            messages=processed_messages,
            stop=stop,
            **kwargs
        )

    def stream_generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> StreamingResponse:
        # 删除 temperature 防止重复传入（初始化已经传入了）
        kwargs.pop("temperature", None)

        def stream_chunks():
            try:
                for chunk in self._stream(messages, stop=stop, **kwargs):
                    yield f"data:{json.dumps({'text': chunk.text}, ensure_ascii=False)}\n\n"
                yield "data:[DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            content=stream_chunks(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Stream-Type": "text-event-stream"
            }
        )

    def _process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """消息预处理（添加系统提示/格式校验）"""
        processed = []
        # 优先添加系统提示
        if self.system_prompt:
            processed.append(SystemMessage(content=self.system_prompt))
        # 简单拼接消息内容，带角色提示，或者根据需要调整格式
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                processed.append(msg)  # 直接保留原始消息对象
            else:
                processed.append(HumanMessage(content=str(msg.content)))  # 默认转为用户消息
        return processed

    @property
    def _llm_type(self) -> str:
        return "ollama-chat-llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model_name, "temperature": self.temperature}
