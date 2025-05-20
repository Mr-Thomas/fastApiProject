from typing import Type, Dict
from langchain_core.language_models import BaseChatModel

# 全局注册表
_llm_registry: Dict[str, Type[BaseChatModel]] = {}


def register_llm(name: str):
    """
    装饰器：注册 LLM 类
    """

    def decorator(cls: Type[BaseChatModel]):
        _llm_registry[name.lower()] = cls
        return cls

    return decorator


def get_llm(name: str, **kwargs) -> BaseChatModel:
    """
    根据名称获取对应 LLM 实例，传递初始化参数
    """
    cls = _llm_registry.get(name.lower())
    if cls is None:
        raise ValueError(f"Unsupported LLM: {name}")
    return cls(**kwargs)
