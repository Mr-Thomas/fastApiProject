import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FastAPI App"
    debug: bool = True

    env: str = "dev"

    # 添加 ollama_url 属性
    ollama_url: str = Field("", env="OLLAMA_URL")
    # 智谱 API 访问密钥配置
    zhipuai_api_key: str = Field("", env="ZHIPUAI_API_KEY")
    # 通义千文 API 访问密钥配置
    dashscope_api_key: str = Field("", env="DASHSCOPE_API_KEY")

    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'dev')}",
        extra='ignore'  # 允许额外的输入配置项，忽略未定义的配置项
    )


settings = Settings()
