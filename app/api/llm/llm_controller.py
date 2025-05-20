from fastapi import APIRouter
from app.schemas.llm_schemas import GenerateTextRequest
from app.services.llm_service_factory import LLMServiceFactory
from app.schemas.base_response import BaseResponse

router = APIRouter()


@router.post("/generate")
def generate_text(request: GenerateTextRequest, stream: bool = False):
    prompt = request.prompt
    model_name = request.model_name
    service_name = request.service_name
    service = LLMServiceFactory.create_service(service_name)
    result = service.generate(prompt, model_name, stream=stream)
    return BaseResponse(code=0, message="Success", data={"generated_text": result})
