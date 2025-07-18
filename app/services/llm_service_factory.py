from typing import Dict, Type
from app.services.llm_interface import LLMInterface


class LLMServiceFactory:
    """大语言模型服务工厂类，用于根据模型名称创建对应的服务实例。"""
    _services: Dict[str, Type[LLMInterface]] = {}

    @classmethod
    def register(cls, service_name: str, service_class: Type[LLMInterface]):
        """
        注册大语言模型服务类。

        :param service_name: 服务名称
        :param service_class: 服务类
        """
        cls._services[service_name] = service_class
        print(f"Registered service: {service_name} -> {service_class.__name__}")    

    @classmethod
    def auto_register(cls, service_name: str):
        """
        类方法的第一个参数是 cls,代表类对象本身(即这个类)

        装饰器：用于自动注册服务类。
        用法：@LLMServiceFactory.auto_register("service_name")
        """
        def decorator(service_class: Type[LLMInterface]):
            """ 
            装饰器函数，用于注册服务类。
            service_class 是被装饰的类对象，由 Python 装饰器机制自动传入
            """
            cls.register(service_name, service_class)
            return service_class
        return decorator

    @classmethod
    def create_service(cls, service_name: str, **kwargs) -> LLMInterface:
        """
        根据模型名称创建对应的服务实例。

        :param service_name: 服务名称
        :param kwargs: 传递给服务类构造函数的参数
        :return: 服务实例
        """
        service_class = cls._services.get(service_name)
        if not service_class:
            raise ValueError(f"Unsupported service: {service_name}")
        return service_class(**kwargs)
