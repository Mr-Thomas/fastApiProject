from http import HTTPStatus
from typing import List, Optional, Any, Mapping
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration,
)
from langchain.chat_models.base import BaseChatModel
from dashscope import Generation
from pydantic import Field
from app.core.config import settings

from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("tongyi")
class TongyiAILLM(BaseChatModel):
    api_key: Optional[str] = None
    model_name: str = Field("qwen-plus", description="Model name")
    temperature: float = Field(0.75, description="Temperature for sampling")

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name or "qwen-plus"
        # 提取获取 API key 的逻辑到单独方法
        self.api_key = self._get_api_key(api_key)

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

        response = Generation.call(**payload)

        if is_stream:
            # 流式响应处理
            full_content = ""
            for partial_response in response:
                if partial_response.status_code != HTTPStatus.OK:
                    raise BizException(code=partial_response.status_code, message=partial_response.message)
                print(partial_response.output.text, end='', flush=True)
                full_content += partial_response.output.text
            content = full_content
        else:
            if response.status_code != HTTPStatus.OK:
                raise BizException(code=response.status_code, message=response.message)
            content = response.output.text

        generation = ChatGeneration(message=AIMessage(content=content), generation_info={"model": self.model_name})
        return ChatResult(generations=[generation])

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
