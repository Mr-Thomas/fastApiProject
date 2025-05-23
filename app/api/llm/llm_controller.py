from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.core.exceptions import BizException
from app.core.logger import logger
from app.schemas.llm_schemas import GenerateTextRequest
from app.services.llm_service_factory import LLMServiceFactory
from app.schemas.base_response import BaseResponse

router = APIRouter()


@router.post("/generate")
def generate_text(request: GenerateTextRequest, stream: bool = Query(False, description="是否使用流式返回"),
                  temperature: float = Query(0.75, ge=0.0, le=1.0, description="采样温度")):
    prompt = request.prompt
    model_name = request.model_name
    service_name = request.service_name
    try:
        service = LLMServiceFactory.create_service(service_name)
        result = service.generate(prompt, model_name, stream=stream, temperature=temperature)
        if isinstance(result, StreamingResponse):
            return result
        return BaseResponse(code=0, message="Success", data={"text": result})
    except Exception as e:
        logger.error(msg=f"service_name:{service_name};model_name:{model_name}", exc_info=e)
        raise BizException(message=f"service_name:{service_name};model_name:{model_name}: {str(e)}") from e
