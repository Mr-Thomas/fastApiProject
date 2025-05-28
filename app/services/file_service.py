import base64
import io
import json
import os
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from fastapi import UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from rapidocr_onnxruntime import RapidOCR
from docx import Document
from app.core.config import settings
from app.core.exceptions import BizException
from app.schemas.llm_schemas import JudgementInfo, Person
from app.services.llm_registry import get_llm
from app.splitters.text_splitter import KeywordExtractor
from app.utils.clean_llm_output import extract_json_block, blocks_to_markdown_html


class FileService:

    def __init__(self, ocr_engine: RapidOCR):
        self.ocr_engine = ocr_engine

    async def convert_image_to_markdown(self, file: UploadFile):
        # content = await file.read()

        # images = convert_pdf_to_images(content, output_dir="D:\\pyWorkspace\\fastApiProject\\app\\documents")

        image_path_0 = os.path.abspath("D:/pyWorkspace/fastApiProject/app/documents/0.png")
        base_64_0 = base64.b64encode(open(image_path_0, 'rb').read()).decode()
        image_path_1 = os.path.abspath("D:/pyWorkspace/fastApiProject/app/documents/1.png")
        base_64_1 = base64.b64encode(open(image_path_1, 'rb').read()).decode()

        llm = get_llm("tongyi", model_name="qwen2.5-vl-32b-instruct")
        print(llm.model_name)
        user_content = [
            {"type": "image", "image": f"data:image/png;base64,{base_64_0}"},
            {"type": "image", "image": f"data:image/png;base64,{base_64_1}"},
            # {"type": "image", "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"},
            # {"type": "image", "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/rabbit.png"},
            # {"type": "text", "text": "请提取该图像的文档结构，并以结构化 JSON 格式返回。"}
            {"type": "text", "text": "请提取该图像的文档结构，并以结构化 HTML 格式返回。"}
        ]
        system_content = [
            {"type": "text", "text": """
                    你是一个专业的文档结构化助手，请根据图像识别并转换为结构化内容。
                    输出要求：
                    - 返回完整 HTML 结构，使用 <h1>-<h3> 标签表示标题
                    - 使用 <p> 表示段落，<ul>/<li> 表示列表
                    - 表格请转换为标准 HTML 表格
                    不要包含任何说明文字，只返回 <body> 中的有效 HTML 内容。
        """}
        ]
        messages = [HumanMessage(content=user_content), SystemMessage(content=system_content)]

        try:
            outputs = llm.invoke(messages)
            markdown_text = outputs.content.strip()
        except Exception as e:
            print(f"Error processing: {e}")
            markdown_text = "Conversion failed due to processing error."

        json_text = extract_json_block(markdown_text)
        blocks = json.loads(json_text)

        md, html = blocks_to_markdown_html(blocks)
        print("Markdown:\n", md)
        print("\nHTML:\n", html)
        output_md_path = "../documents/output.md"
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved Markdown to: {output_md_path}")

        return md, html

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
        extractor = KeywordExtractor(llm, model_path="/app/local_models/bge-small-zh")
        chunks = extractor.get_chunks(text=text)
        return chunks

    async def format_document(self, text: str) -> Dict[str, Any]:
        # llm = get_llm("ollama", model_name="modelscope.cn/Qwen/QwQ-32B-GGUF:latest", temperature=0)
        llm = get_llm("tongyi", model_name="qwen-plus", temperature=0)
        print(llm.temperature, llm.model_name)
        extractor = KeywordExtractor(llm, model_path="D:\\pyWorkspace\\fastApiProject\\app\\local_models\\bge-small-zh")
        # format = extractor.extract_from_text_by_model(text=text, model_cls=JudgementInfo)
        format = extractor.extract_whole_text_by_model(text=text, model_cls=JudgementInfo)
        return format
