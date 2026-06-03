import json
from typing import Optional, List, Any, Mapping, Iterator
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain.chat_models.base import BaseChatModel
from pydantic import Field, PrivateAttr
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("deepseek")
class DeepseekLLM(BaseChatModel):
    """基于 OpenAI SDK 封装的 Deepseek Chat Model，兼容 LangChain 聊天接口。"""
    model_name: str = Field(default="deepseek-v4-flash", description="模型名称")
    temperature: float = Field(default=1.0, description="采样温度")
    max_tokens: int = Field(default=8192, description="最大输出 token 数")
    top_p: float = Field(default=1.0, description="核采样参数")
    _client: OpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("model_name", "deepseek-v4-flash")
        self.temperature = kwargs.get("temperature", 1.0)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.top_p = kwargs.get("top_p", 1.0)

        api_key = settings.deepseek_api_key
        if not api_key:
            raise BizException(message="DEEPSEEK_API_KEY 未配置")
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        try:
            payload = self._build_payload(messages, stop, **kwargs)
            response = self._client.chat.completions.create(**payload)

            if not response or not response.choices:
                raise BizException(message="Deepseek 返回为空")

            message = response.choices[0].message
            content = message.content or ""

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=content,
                            additional_kwargs={"reasoning_content": message.reasoning_content or ""}
                        ),
                        generation_info={"model": self.model_name}
                    )
                ]
            )
        except Exception as e:
            raise BizException(message=f"Deepseek 生成失败: {str(e)}") from e

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, stream=True, **kwargs)
        stream: Stream[ChatCompletionChunk] = self._client.chat.completions.create(**payload)

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            reasoning_content = getattr(delta, "reasoning_content", None) or ""
            if content:
                yield ChatGenerationChunk(
                    text=content,
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs={"reasoning_content": reasoning_content}
                    )
                )

    def stream_generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> StreamingResponse:
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

    def _build_payload(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            stream: bool = False,
            **kwargs: Any,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        return payload

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """LangChain 消息格式转为 Deepseek API 所需格式。"""
        result = []
        for msg in messages:
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            else:
                role = "user"
            result.append({"role": role, "content": msg.content})
        return result

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
