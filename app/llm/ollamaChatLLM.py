import json
from typing import List, Optional, Any, Mapping
from langchain.schema import AIMessage, BaseMessage, ChatResult, ChatGeneration
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import GenerationChunk
from langchain_ollama import OllamaLLM
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

    def __init__(self, *, model_name: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        if base_url is None:
            base_url = settings.ollama_url
            if not base_url:
                raise BizException(message="OLLAMA_URL 未配置")
        self.temperature = kwargs.pop("temperature", 0.8)
        self._ollama_llm = OllamaLLM(model=self.model_name, base_url=base_url, temperature=self.temperature, **kwargs)
        print(self._ollama_llm.temperature)
        print(self._ollama_llm.model)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)

        # 删除 temperature 防止重复传入（初始化已经传入了）
        kwargs.pop("temperature", None)
        kwargs.pop("stream", None)

        llm_result = self._ollama_llm._generate([prompt], stop=stop, **kwargs)
        full_text = llm_result.generations[0][0].text
        generation = ChatGeneration(message=AIMessage(content=full_text))
        return ChatResult(generations=[generation])

    def stream_generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> StreamingResponse:
        prompt = self._messages_to_prompt(messages)
        # 删除 temperature 防止重复传入（初始化已经传入了）
        kwargs.pop("temperature", None)
        kwargs.pop("stream", None)
        chunks: List[GenerationChunk] = list(self._ollama_llm._stream(prompt, stop=stop, **kwargs))

        def stream_chunks():
            try:
                for chunk in chunks:
                    yield f"data:{json.dumps({'text': chunk.text}, ensure_ascii=False)}\n\n"
                yield f"data:[DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            content=stream_chunks(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Stream-Type": "text-event-stream"
            }
        )

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        # 仅支持单一 user message，以防多轮误导模型结构输出
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            return messages[0].content
        prompt = ""
        # 简单拼接消息内容，带角色提示，或者根据需要调整格式
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
