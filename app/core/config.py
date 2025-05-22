import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="FastAPI App", validation_alias="APP_NAME")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")
    env: str = Field(default="dev", validation_alias="ENV")
    api_prefix: str = Field(default="/api", validation_alias="API_PREFIX")

    # 添加 ollama_url 属性
    ollama_url: str = Field(default="", validation_alias="OLLAMA_URL")
    # 智谱 API 访问密钥配置
    zhipuai_api_key: str = Field(default="", validation_alias="ZHIPUAI_API_KEY")
    # 通义千文 API 访问密钥配置
    dashscope_api_key: str = Field(default="", validation_alias="DASHSCOPE_API_KEY")

    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'dev')}",
        extra='ignore'  # 允许额外的输入配置项，忽略未定义的配置项
    )


settings = Settings()
