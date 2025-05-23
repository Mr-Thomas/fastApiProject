from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="prompt不能为空！")
    service_name: str = Field("ollama", description="服务名称，默认为 ollama")
    model_name: str = Field("deepseek-r1:1.5b", description="模型名称，默认为 deepseek-r1:1.5b")


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


class PartyInfo(BaseModel):
    role: Optional[str] = Field(None, description="角色（原告、被告等）")
    name: Optional[str] = Field(None, description="姓名")
    gender: Optional[str] = Field(None, description="性别")
    date_of_birth: Optional[str] = Field(None, description="出生日期")
    ethnic_group: Optional[str] = Field(None, description="民族（如汉族）")
    address: Optional[str] = Field(None, description="住址")
    law_firm_name: Optional[str] = Field(None, description="代理律师所属律所名称")
    law_firm_address: Optional[str] = Field(None, description="律所地址")


class Judge(BaseModel):
    role: Optional[str] = Field(None, description="角色（审判员、审判长等）")
    name: Optional[str] = Field(None, description="姓名")

    @field_validator("role", "name", mode="before")
    @classmethod
    def flatten_list_if_needed(cls, v: Union[str, list, None]):
        if v is None:
            return None
        # 如果是字符串直接返回
        if isinstance(v, str):
            return v.strip()
        # 如果是列表，拼接为字符串（中文顿号“、”）
        if isinstance(v, list) and all(isinstance(item, str) for item in v):
            return "、".join(item.strip() for item in v if item.strip())
        return v  # 非字符串或字符串列表，保持原样


class Evidence(BaseModel):
    party: Optional[str] = Field(None, description="证据提交方（原告或被告）")
    name: Optional[str] = Field(None, description="证据名称")
    purpose: Optional[str] = Field(None, description="证明目的")
    court_opinion: Optional[str] = Field(None, description="法院认定")

    @field_validator('party', 'name', 'purpose', 'court_opinion', mode='before')
    @classmethod
    def validate_other_info(cls, v: Union[str, List[str], None]) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            cleaned = [item.strip() for item in v if isinstance(item, str) and item.strip()]
            return "、".join(cleaned) if cleaned else None
        return str(v).strip()  # 最后一层兜底防御


class JudgementInfo(BaseModel):
    court_name: Optional[str] = Field(None, description="法院名称")
    court_code: Optional[str] = Field(None, description="案件编号")
    cause_of_action: Optional[str] = Field(None, description="案由（案子类型）")
    filing_date: Optional[str] = Field(None, description="立案时间")
    trial_procedure: Optional[str] = Field(None, description="审理程序（如简易程序）")
    judgment_date: Optional[str] = Field(None, description="结案时间（判决时间）")
    judges: List[Judge] = Field(None, description="审判人员信息列表")
    parties: List[PartyInfo] = Field(None, description="当事人信息列表")
    plaintiff_claims: List[str] = Field(default_factory=list, description="原告诉讼请求")
    facts_and_reasons: List[str] = Field(default_factory=list, description="原告事实与理由")
    defendant_defense: List[str] = Field(default_factory=list, description="被告答辩")
    evidences: List[Evidence] = Field(None, description="证据清单及认定")
    facts: List[str] = Field(default_factory=list, description="法院查明的事实")
    findings: List[str] = Field(default_factory=list, description="法院认定意见或结论")
    legal_basis: List[str] = Field(default_factory=list, description="法律依据（法律条文）")
    judgment_result: List[str] = Field(default_factory=list, description="判决结果")
    other_info: List[str] = Field(default_factory=list, description="程序性信息（上诉权利）")

    @field_validator("court_name", "court_code", "cause_of_action", "filing_date", "trial_procedure",
                     "judgment_date", mode="before")
    @classmethod
    def clean_judgment_date(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None  # 空字符串也转 None

    @field_validator('plaintiff_claims', 'facts_and_reasons', 'defendant_defense', 'facts', 'findings',
                     'legal_basis', 'judgment_result', 'other_info', mode='before')
    @classmethod
    def validate_other_info(cls, v: Union[str, List[str], None]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator('judges', 'parties', 'evidences', mode='before')
    @classmethod
    def validate_evidences(cls, v: Union[dict, List[dict], None]) -> List[dict]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return list(v)
