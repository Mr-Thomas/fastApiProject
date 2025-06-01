from langchain_core.language_models import BaseChatModel
from pydantic import Field, PrivateAttr
from typing import Optional, List, Any, Mapping
from zhipuai import ZhipuAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app.core.config import settings
from app.core.exceptions import BizException
from app.services.llm_registry import register_llm


@register_llm("zhipuai")
class ZhipuAILLM(BaseChatModel):
    model_name: str = Field(default="glm-4-plus", description="ZhipuAI 模型名称")
    temperature: float = Field(default=0.75, description="生成温度")
    # client 不参与 pydantic 序列化
    _client: ZhipuAI = PrivateAttr()

    def __init__(self, client: Optional[ZhipuAI] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 **kwargs: Any, ):
        super().__init__(**kwargs)
        self.model_name = model_name or "glm-4-plus"
        self.temperature = temperature or 0.75
        if client:
            self._client = client
        else:
            zhipuai_api_key = settings.zhipuai_api_key
            if not zhipuai_api_key:
                raise BizException(message="zhipuai_api_key 未配置")
            self._client = ZhipuAI(api_key=zhipuai_api_key)  # 初始化 client

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        try:
            payload = {
                "model": self.model_name,
                "messages": self._convert_messages(messages),
                "temperature": self.temperature,
            }
            if stop:
                payload["stop"] = stop

            response = self._client.chat.completions.create(**payload)

            if not response or not response.choices:
                raise BizException(message="ZhipuAI 返回为空")

            message = response.choices[0].message
            content = message.content if message else ""

            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=content), generation_info={"model": self.model_name})]
            )
        except Exception as e:
            raise BizException(message=f"ZhipuAI 生成失败: {str(e)}") from e

    @property
    def _llm_type(self) -> str:
        return "zhipuai-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        LangChain 消息格式 -> ZhipuAI 所需格式
        """
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
