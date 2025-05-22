from fastapi import FastAPI
from app.api.llm import llm_controller, file_controller
from app.core.config import settings
from app.core.logger import logger


def register_routers(app: FastAPI):
    app.include_router(llm_controller.router, prefix="/api")
    app.include_router(file_controller.router, prefix="/api")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.0.1",
        debug=settings.debug,
        description="This is a demo project for FastAPI",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    register_routers(app)

    # 启动日志
    logger.info("🚀 FastAPI app created successfully!")

    return app


app = create_app()
