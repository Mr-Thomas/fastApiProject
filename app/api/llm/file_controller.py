from fastapi import APIRouter, UploadFile, File, Depends
from rapidocr_onnxruntime import RapidOCR

from app.schemas.base_response import BaseResponse
from app.services.file_service import FileService

router = APIRouter()

# 单例 OCR 引擎实例（可全局使用）
ocr_engine = RapidOCR()


def get_file_service():
    return FileService(ocr_engine=ocr_engine)


@router.post("/ocr")
async def ocr_upload(file: UploadFile = File(...), service: FileService = Depends(get_file_service)):
    ocr_text = await service.process_file(file)
    return BaseResponse(code=0, message="Success", data={"generated_text": ocr_text})


@router.post("/ocr_split")
async def ocr_upload(file: UploadFile = File(...), service: FileService = Depends(get_file_service)):
    ocr_text = await service.process_file(file)
    chunks = await service.chunk_document(ocr_text)
    return BaseResponse(code=0, message="Success", data={"chunks": chunks})


@router.post("/ocr_split_format")
async def ocr_upload(file: UploadFile = File(...), service: FileService = Depends(get_file_service)):
    ocr_text = await service.process_file(file)
    format = await service.format_document(ocr_text)
    return BaseResponse(code=0, message="Success", data={"format": format})


@router.post("/image_to_markdown")
async def ocr_upload(file: UploadFile = File(...), service: FileService = Depends(get_file_service)):
    # todo 待完善，图片转markdown或者HTML
    md, html = await service.convert_image_to_markdown(file)
    return BaseResponse(code=0, message="Success", data={"md": md, "html": html})
