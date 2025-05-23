import json
from http import HTTPStatus
from typing import List, Optional, Any, Mapping, Union, Generator

from dashscope.api_entities.dashscope_response import GenerationResponse
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration,
)
from langchain.chat_models.base import BaseChatModel
from dashscope import Generation
from pydantic import Field
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("tongyi")
class TongyiAILLM(BaseChatModel):
    api_key: Optional[str] = None
    model_name: str = Field("qwen-plus", description="Model name")
    temperature: float = Field(0.75, description="Temperature for sampling")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("model_name", "qwen-plus")
        self.api_key = self._get_api_key(kwargs.get("api_key"))

    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        """
        获取并验证 dashscope API key。
        如果传入的 api_key 为空，则尝试从配置中获取。
        如果配置中也未找到，则抛出异常。
        """
        if api_key:
            return api_key
        api_key = settings.dashscope_api_key
        if not api_key:
            raise BizException(message="dashscope_api_key 未配置")
        return api_key

    @property
    def _llm_type(self) -> str:
        return "tongyi-llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        try:
            payload = {
                "api_key": self.api_key,
                "model": self.model_name,
                "messages": self._convert_messages(messages),
                "temperature": self.temperature,
                **kwargs,
            }
            if stop:
                payload["stop"] = stop

            response = Generation.call(**payload)

            if response.status_code != HTTPStatus.OK:
                raise BizException(code=response.status_code, message=response.message)
            content = response.output.text
            generation = ChatGeneration(message=AIMessage(content=content),
                                        generation_info={"model": self.model_name})
            return ChatResult(generations=[generation])

        except Exception as e:
            raise BizException(message=f"[TongyiAILLM_generate] 调用失败: {str(e)}") from e

    def stream_generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> StreamingResponse:
        """处理流式响应（FastAPI 专用）"""
        payload = {
            "api_key": self.api_key,
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            **kwargs,
        }
        if stop:
            payload["stop"] = stop

        is_stream = kwargs.get('stream', False)
        # 增量式流式输出【默认false】
        payload["incremental_output"] = is_stream

        response = Generation.call(**payload)
        return StreamingResponse(
            content=self._stream_resp(response),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Stream-Type": "text-event-stream"
            }
        )

    def _stream_resp(
            self,
            response: Union[GenerationResponse, Generator[GenerationResponse, None, None]]
    ) -> Generator[str, None, None]:
        try:
            for partial_response in response:
                if partial_response.status_code != HTTPStatus.OK:
                    raise BizException(code=partial_response.status_code, message=partial_response.message)
                # SSE 格式的事件数据，两个换行表示事件结束
                yield f"data:{json.dumps({'text': partial_response.output.text}, ensure_ascii=False)}\n\n"
            # 发送完整内容，结束标记
            yield f"data:[DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        LangChain 消息格式转为通义千文 Generation.call 所需格式
        """
        converted = []
        for msg in messages:
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            else:
                role = "user"
            converted.append({"role": role, "content": msg.content})
        return converted
