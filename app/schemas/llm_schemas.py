from typing import Optional, List
from pydantic import BaseModel, Field

from app.utils.document_process import LegalDocumentExtractor


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


class PartyInfo(BaseModel):
    role: str = Field(None, description="角色（原告、被告等）")
    name: str = Field(None, description="姓名")
    gender: str = Field(None, description="性别")
    date_of_birth: str = Field(None, description="出生日期")
    ethnic_group: str = Field(None, description="民族（如汉族）")
    address: str = Field(None, description="住址")
    law_firm_name: str = Field(None, description="代理律师所属律所名称")
    law_firm_address: str = Field(None, description="律所地址")


class Judge(BaseModel):
    role: str = Field(None, description="角色（审判员、书记员等）")
    name: str = Field(None, description="姓名")


class ClaimDefense(BaseModel):
    plaintiff_claims: List[str] = Field(None, description="原告诉讼请求")
    facts_and_reasons: List[str] = Field(None, description="原告事实与理由")
    defendant_defense: List[str] = Field(None, description="被告答辩")


class Evidence(BaseModel):
    party: str = Field(None, description="证据提交方（原告或被告）")
    name: str = Field(None, description="证据名称")
    purpose: str = Field(None, description="证明目的")
    court_opinion: str = Field(None, description="法院认定")


class FactFinding(BaseModel):
    fact: List[str] = Field(None, description="法院查明的事实")
    finding: List[str] = Field(None, description="法院认定意见或结论")


class JudgmentBasis(BaseModel):
    legal_basis: List[str] = Field(None, description="法律依据（法律条文）")
    judgment_result: List[str] = Field(None, description="判决结果")


class JudgementInfo(BaseModel):
    # 一、案件基本信息
    court_name: str = Field(None, description="法院名称")
    court_code: str = Field(None, description="案件编号")
    cause_of_action: str = Field(None, description="案由（案子类型）")
    filing_date: str = Field(None, description="立案时间")
    trial_procedure: str = Field(None, description="审理程序（如简易程序）")
    judgment_date: str = Field(None, description="结案时间（判决时间）")

    judges: List[Judge] = Field(default=None, description="审判人员、书记员信息列表")

    # 二、当事人信息
    parties: List[PartyInfo] = Field(None, description="当事人信息列表")

    # 三、诉讼请求与答辩意见
    claim_defense: ClaimDefense = Field(None, description="诉讼答辩")

    # 四、证据清单及认定
    evidences: List[Evidence] = Field(None, description="证据清单及认定")

    # 五、法院认定事实
    fact_findings: List[FactFinding] = Field(None, description="法院查明事实与认定意见")

    # 六、判决依据及判决内容
    legal_judgment: JudgmentBasis = Field(None, description="判决依据及判决内容(法律依据、判决情况)")

    # 七、程序性告知
    other_info: List[str] = Field(None, description="程序性信息（上诉权利）")
