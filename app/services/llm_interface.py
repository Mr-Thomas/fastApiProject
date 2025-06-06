from abc import ABC, abstractmethod
from typing import Union
from fastapi.responses import StreamingResponse


class LLMInterface(ABC):
    """大语言模型接口的抽象基类。"""

    @abstractmethod
    def generate(self, prompt: str, model_name: str, **kwargs) -> Union[str, StreamingResponse]:
        """
        根据输入的提示词生成文本。

        :param prompt: 输入的提示词
        :param model_name: 模型名称
        :return: 生成的文本
        """
        pass
