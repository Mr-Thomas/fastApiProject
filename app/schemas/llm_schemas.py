from typing import Optional
from pydantic import BaseModel, Field


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="prompt不能为空！")
    service_name: str = "ollama"
    model_name: str = "deepseek-r1:1.5b"


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    plaintiff_name: Optional[str] = Field(default=None, description="原告姓名")
    plaintiff_address: Optional[str] = Field(default=None, description="原告地址")
    defendant_name: Optional[str] = Field(default=None, description="被告姓名")
    defendant_address: Optional[str] = Field(default=None, description="被告地址")
    cell_name: Optional[str] = Field(default=None, description="小区名称")
    overdue_charge: Optional[str] = Field(default=None, description="逾期费用")
    property_fee: Optional[str] = Field(default=None, description="物业费")
