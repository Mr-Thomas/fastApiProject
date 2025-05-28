import fitz  # PyMuPDF
from typing import Union, List

from app.core.exceptions import BizException

import json
from docx import Document
from pathlib import Path


def doc_to_labelstudio(docx_path, task_type="ner"):
    """
    将DOCX转换为Label Studio兼容的JSON格式
    :param docx_path: 输入文件路径
    :param task_type: 任务类型 (ner|text_classification)
    :return: Label Studio格式的字典
    """
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    # Label Studio基础模板
    ls_template = {
        "data": {
            "text": text,
            "file_name": Path(docx_path).name
        },
        "annotations": [{
            "result": [],
            "ground_truth": False,
            "model_version": None
        }]
    }

    # 根据不同任务类型调整结构
    if task_type == "text_classification":
        ls_template["annotations"][0]["result"] = [{
            "type": "choices",
            "value": {"choices": []}  # 等待标注的分类标签
        }]

    return ls_template


if __name__ == '__main__':
    # 配置路径
    input_path = Path(
        "D:\\pyWorkspace\\fastApiProject\\app\\documents\\doc_35803_离婚纠纷_（2022）豫1628民初6633号_判决书原文.docx")  # 替换为你的文件路径
    output_json = "../documents/label_studio_import.json"

    # 转换并保存
    result = doc_to_labelstudio(input_path, task_type="ner")  # 或 text_classification

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump([result], f, ensure_ascii=False, indent=2)

    print(f"Label Studio JSON 已生成: {output_json}")


def extract_embedded_images(
        pdf_input: Union[str, Path, bytes],
        output_dir: Union[str, Path],
        prefix: str = "img"
) -> List[str]:
    """
    从 PDF 文件（路径或 bytes）中提取嵌入图像。

    Args:
        pdf_input (str | Path | bytes): PDF 路径或二进制内容。
        output_dir (str | Path): 图像输出目录。
        prefix (str): 图像文件名前缀。

    Returns:
        List[str]: 图像文件的保存路径列表。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 支持路径 / 内存流 ===
    try:
        if isinstance(pdf_input, (str, Path)):
            doc = fitz.open(pdf_input)
        elif isinstance(pdf_input, bytes):
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        else:
            raise BizException(message="pdf_input必须是路径或bytes")
    except Exception as e:
        raise BizException(message=f"无法打开 PDF: {e}")from e

    image_paths = []
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"].lower()
                filename = f"{prefix}_page{page_num}_img{img_index}.{ext}"
                img_path = output_dir / filename
                img_path.write_bytes(image_bytes)
                image_paths.append(str(img_path))
            except Exception as e:
                print(f"⚠️ 第 {page_num} 页图像 {img_index} 提取失败: {e}")
                continue

    return image_paths


def convert_pdf_to_images(
        pdf_input: Union[str, Path, bytes],
        output_dir: Union[str, Path],
        image_format: str = "png",
        dpi: int = 300
) -> List[str]:
    """
    将 PDF 转换为图像，每页一张，保存到指定目录。

    Args:
        pdf_input (str | Path | bytes): PDF 路径或二进制内容。
        output_dir (str | Path): 图片输出目录。
        image_format (str): 图片格式（"png" 或 "jpg"）。
        dpi (int): 图片分辨率，默认 300。

    Returns:
        List[str]: 每页图像文件的保存路径。
    """
    if isinstance(pdf_input, (str, Path)):
        doc = fitz.open(pdf_input)
    elif isinstance(pdf_input, bytes):
        doc = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        raise BizException(message="pdf_input必须是文件路径或PDF二进制内容")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        image_path = output_dir / f"{page_number + 1}_page.{image_format}"
        pix.save(str(image_path))
        image_paths.append(str(image_path))

    return image_paths
