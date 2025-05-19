from rapidocr_onnxruntime import RapidOCR
import os


# 加载jpg文件
def load_jpg_file(jpg_file: str):
    work_dir = "../documents"
    ocr = RapidOCR()
    result, _ = ocr(os.path.join(work_dir, jpg_file))
    docs = ""
    if result:
        # 从OCR结果中提取文本信息，line[1]表示每行识别结果的文本部分
        ocr_result = [line[1] for line in result]
        docs += docs.join(ocr_result)
    return docs


def load_jpg_file_list(jpg_files: list):
    work_dir = "../documents"
    ocr = RapidOCR()
    docs = []
    for jpg_file in jpg_files:
        result, _ = ocr(os.path.join(work_dir, jpg_file))
        if result:
            # 从OCR结果中提取文本信息，line[1]表示每行识别结果的文本部分
            ocr_result = [line[1] for line in result]
            docs.extend(ocr_result)
    return "".join(docs)
