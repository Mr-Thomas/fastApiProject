import io
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from fastapi import UploadFile
from langchain_ollama import OllamaLLM
from rapidocr_onnxruntime import RapidOCR
from docx import Document
from app.core.config import settings
from app.core.exceptions import BizException
from app.schemas.llm_schemas import JudgementInfo, Person
from app.services.llm_registry import get_llm
from app.splitters.text_splitter import KeywordExtractor


class FileService:

    def __init__(self, ocr_engine: RapidOCR):
        self.ocr_engine = ocr_engine

    async def process_file(self, file: UploadFile) -> str:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            return await self._extract_from_pdf(file)
        elif filename.endswith(".docx"):
            return await self._extract_from_docx(file)
        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            return await self._extract_from_image(file)
        else:
            raise BizException(message="Unsupported file type")

    async def _extract_from_image(self, file: UploadFile) -> str:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # 预处理：增强对比度和锐化，帮助OCR识别印章文字
        enhancer = ImageEnhance.Contrast(image)
        image_enhanced = enhancer.enhance(2.0)  # 增强对比度，系数可调
        image_enhanced = image_enhanced.filter(ImageFilter.SHARPEN)  # 锐化
        # PIL Image 转 numpy ndarray (RGB)
        image_np = np.array(image_enhanced)
        # 转成 BGR，适配 RapidOCR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result, _ = self.ocr_engine(image_bgr)
        ocr_result = [item[1] for item in result if item[1].strip() != ""]  # 过滤空结果
        return "\n".join(ocr_result)

    async def _extract_from_docx(self, file: UploadFile) -> str:
        contents = await file.read()
        doc = Document(io.BytesIO(contents))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    async def _extract_from_pdf(self, file: UploadFile) -> str:
        contents = await file.read()
        with fitz.open(stream=contents, filetype="pdf") as doc:
            # 自动判断 PDF 是否为文本型
            has_text_layer = any(page.get_text().strip() for page in doc)

            if has_text_layer:
                # 文本型 PDF，直接提取更快
                return "\n".join(page.get_text().strip() for page in doc)
            else:
                # 扫描型 PDF，逐页转图片后 OCR
                zoom = 2  # 2倍放大，提升分辨率（效果约等于 144dpi）
                matrix = fitz.Matrix(zoom, zoom)
                all_text = []
                for page in doc:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    result, _ = self.ocr_engine(img)
                    page_text = "\n".join([item[1] for item in result])
                    all_text.append(page_text)
                return "\n".join(all_text)

    async def chunk_document(self, text: str) -> List[str]:
        # llm = get_llm("tongyi", model="qwen-plus-latest")
        llm = OllamaLLM(base_url=settings.ollama_url, model="deepseek-r1:14b")
        extractor = KeywordExtractor(llm, model_path="D:/pyWorkspace/fastApiProject/app/models/bge-small-zh")
        chunks = extractor.get_chunks(text=text)
        return chunks

    async def format_document(self, text: str) -> Dict[str, Any]:
        # llm = get_llm("ollama", model_name="modelscope.cn/Qwen/QwQ-32B-GGUF:latest", temperature=0)
        llm = get_llm("tongyi", model_name="qwen-plus", temperature=0)
        print(llm.temperature, llm.model_name)
        extractor = KeywordExtractor(llm, model_path="D:/pyWorkspace/fastApiProject/app/models/bge-small-zh")
        # format = extractor.extract_from_text_by_model(text=text, model_cls=JudgementInfo)
        format = extractor.extract_whole_text_by_model(text=text, model_cls=JudgementInfo)
        return format
