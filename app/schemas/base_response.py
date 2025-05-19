from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    code: int
    message: str
    data: T | None = None
