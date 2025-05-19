from fastapi import APIRouter, UploadFile, File
from langchain_ollama import OllamaLLM
from rapidocr_onnxruntime import RapidOCR
from app.core.config import settings
from app.llm.tongyiLLM import TongyiAILLM
from app.schemas.llm_schemas import GenerateTextRequest, Person
from app.services.llm_factory import LLMServiceFactory
from app.schemas.base_response import BaseResponse
from PIL import Image
import io

from app.splitters.text_splitter import SemanticTextSplitter, KeywordExtractor

router = APIRouter()


@router.post("/generate")
def generate_text(request: GenerateTextRequest, stream: bool = False):
    prompt = request.prompt
    model_name = request.model_name
    service_name = request.service_name
    service = LLMServiceFactory.create_service(service_name)
    result = service.generate(prompt, model_name, stream=stream)
    return BaseResponse(code=0, message="Success", data={"generated_text": result})


ocr_engine = RapidOCR()


@router.post("/ocr")
async def ocr_upload(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result, _ = ocr_engine(image)
    ocr_result = [item[1] for item in result]
    return BaseResponse(code=0, message="Success", data={"generated_text": "\n".join(ocr_result)})


@router.post("/ocr_split")
async def ocr_upload(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result, _ = ocr_engine(image)
    ocr_result = [item[1] for item in result]
    text = "\n".join(ocr_result)
    llm = TongyiAILLM(api_key=settings.dashscope_api_key)
    extractor = KeywordExtractor(llm, model_path="D:/pyWorkspace/fastApiProject/app/models/bge-small-zh",
                                 similarity_threshold=0.75)
    chunks = extractor.get_chunks(text)
    return BaseResponse(code=0, message="Success", data={"generated_text": chunks})


@router.post("/ocr_split_format")
async def ocr_upload(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result, _ = ocr_engine(image)
    ocr_result = [item[1] for item in result]
    text = "\n".join(ocr_result)
    llm = TongyiAILLM(api_key=settings.dashscope_api_key, model="qwen-plus-latest")
    # llm = OllamaLLM(base_url=settings.ollama_url, model="deepseek-r1:14b")
    extractor = KeywordExtractor(llm, model_path="D:/pyWorkspace/fastApiProject/app/models/bge-small-zh",
                                 similarity_threshold=0.75)
    chunks = extractor.extract_from_text_by_model(model_cls=Person, text=text)
    # extractor = KeywordExtractor(llm)
    # chunks = extractor.extract_whole_text_by_model(text=text, model_cls=Person)
    return BaseResponse(code=0, message="Success", data={"generated_text": chunks})
