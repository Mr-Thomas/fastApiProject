from fastapi import FastAPI
from app.api.router import api_router
from app.core.config import settings
from app.core.logger import logger


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.0.1",
        debug=settings.debug,
        description="This is a demo project for FastAPI",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 注册总路由
    app.include_router(api_router)

    # 启动日志
    logger.info("🚀 FastAPI app created successfully!")

    return app


app = create_app()
