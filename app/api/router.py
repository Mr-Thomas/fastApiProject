from fastapi import APIRouter
from app.api.llm import llm_controller, file_controller
from app.core.config import settings

api_router = APIRouter(prefix=settings.api_prefix)

# 按模块注册子路由
api_router.include_router(llm_controller.router)
api_router.include_router(file_controller.router)
