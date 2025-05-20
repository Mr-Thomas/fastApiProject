from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.exceptions import BizException
from app.schemas.base_response import BaseResponse


async def biz_exception_handler(request: Request, exc: BizException):
    return JSONResponse(
        status_code=200,
        content=BaseResponse[Any](
            code=exc.code,
            message=exc.message,
            data=getattr(exc, 'data', None)
        ).model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = []
    for err in exc.errors():
        # 拼接字段路径，排除第一个元素（通常为 "body"、"query"等）
        loc = err.get("loc", [])
        field_name = ".".join(str(i) for i in loc if i != "body")
        error_details.append({
            "field": field_name,
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        })
    return JSONResponse(
        status_code=200,
        content=BaseResponse[list](
            code=422,
            message="参数校验错误",
            data=error_details
        ).model_dump()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse[Any](
            code=exc.status_code,
            message=exc.detail,
            data=getattr(exc, "data", None)
        ).model_dump()
    )
