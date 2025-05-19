from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import BizException
from app.schemas.base_response import BaseResponse


async def biz_exception_handler(request: Request, exc: BizException):
    return JSONResponse(
        status_code=200,
        content=BaseResponse[None](
            code=exc.code,
            message=exc.message,
            data=None
        ).model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = []
    for error in exc.errors():
        field_name = ".".join(map(str, error["loc"]))
        error_details.append({
            "field": field_name,
            "message": error["msg"],
            "type": error["type"],
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
        content=BaseResponse[None](
            code=exc.status_code,
            message=exc.detail,
            data=None
        ).model_dump()
    )
