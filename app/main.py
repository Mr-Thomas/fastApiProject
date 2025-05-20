from fastapi import FastAPI
from app.api.llm import llm_controller, file_controller
from app.core.config import settings
from app.core.exception_handler import (
    biz_exception_handler,
    validation_exception_handler,
    http_exception_handler,
)
from app.core.exceptions import BizException
from fastapi.exceptions import RequestValidationError, HTTPException

app = FastAPI(title=settings.app_name, version="0.0.1", debug=settings.debug,
              description="This is a demo project for FastAPI",
              docs_url="/docs",
              redoc_url="/redoc",
              reload=settings.debug)

# 注册异常处理器
app.add_exception_handler(BizException, biz_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

# 包含路由
app.include_router(llm_controller.router, prefix="/api")
app.include_router(file_controller.router, prefix="/api")
