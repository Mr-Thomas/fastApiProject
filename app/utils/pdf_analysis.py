import pdfplumber  # pip install pdfplumber
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any


class SchemaExtractor:
    def __init__(self):
        self.case_fields = defaultdict(list)
        self.current_cause = None  # 用于跟踪当前案由

    def clean_table(self, table: List[List[str]]) -> List[List[str]]:
        """针对图片中表格结构的专用清洗方法"""
        cleaned_rows = []

        for row in table:
            # 跳过完全空的行
            if not row or all(cell in ('', ' ') for cell in row):
                continue

            # 格式: [空列, 案由, 字段名, 描述]
            # 直接跳过前导空列
            cells = []
            for idx, cell in enumerate(row):
                if idx == 0 and (cell or '').strip() == '':
                    continue  # 跳过第一个空列
                    # 处理换行：替换为空格，避免破坏语义
                cleaned_cell = (cell or '').replace('\n', ' ').strip()
                cells.append(cleaned_cell)

            # 跳过表头行（包含特定关键词的行）
            if len(cells) >= 2 and any(word in cells[0] for word in ["数据项清单", "案由", "数据库字段"]):
                continue

            # 提取有效列（根据图片调整为索引1-3列）
            if len(cells) == 3:
                cause, field, desc = cells[0], cells[1], cells[2]
            elif len(cells) == 2:  # 可能是跨行情况
                if self.current_cause:
                    field, desc = cells[0], cells[1]
                    cause = self.current_cause
                else:
                    continue
            else:
                continue
            # 更新当前案由（即使后面字段为空）
            if cause:
                self.current_cause = cause

            # 确保字段和描述有效
            if field and desc:
                # 合并可能被拆分的描述（处理跨行换行）
                if cleaned_rows and not field and desc:
                    cleaned_rows[-1][2] += " " + desc  # 表示已处理数据中最后一行的第3列
                else:
                    cleaned_rows.append([cause, field, desc])

        return cleaned_rows

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """将蛇形命名转为驼峰命名"""
        parts = snake_str.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    def extract_fields(self, pdf_path: str) -> None:
        """从PDF中提取字段信息，改进跨页处理"""
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # 提取表格时保留更多空白单元格
                table_settings = {
                    # "lines"：基于实际画出的竖线（边框线）；"edges"：基于图像边缘检测；"explicit"：基于显式的分割线（标准化导出的 PDF（如 Word 导出））；"text"：基于文本对齐（推荐用于无边框 PDF）
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "keep_blank_chars": False,  # 忽略纯空白字符
                    "text_tolerance": 3,  # 增大文本识别容差
                    "intersection_tolerance": 5,  # 提高交叉点检测容差
                }
                tables = page.extract_tables(table_settings)
                for table in tables:
                    # 第一页跳过表头
                    if page_idx == 0 and table:
                        table = table[3:] if len(table) > 3 else table
                    cleaned_rows = self.clean_table(table)
                    for cause, field, desc in cleaned_rows:
                        if field and desc:  # 确保字段和描述都不为空
                            self.case_fields[cause].append((field, desc))

    def generate_schema(self, cause: str, fields: List[Tuple[str, str]]) -> Dict[str, Any]:
        """生成schema结构"""
        return {
            "type": "object",
            "title": f"{cause}要素表",
            "properties": {
                "basicInfo": {
                    "type": "object",
                    "title": "文书基本信息",
                    "properties": {
                        "caseCode": {"type": "string", "title": "案号"},
                        "caseDesc": {"type": "string", "title": "案由"},
                        "trialCourt": {"type": "string", "title": "审理法院"},
                        "trialProcedure": {"type": "string", "title": "审判程序"},
                        "judgmentDate": {"type": "string", "title": "判决日期"}
                    }
                },
                "elementInfo": {
                    "type": "object",
                    "title": f"{cause}要素信息",
                    "properties": {
                        self.to_camel_case(field): {
                            "type": "string",
                            "title": desc.strip(),
                        } for field, desc in fields
                    }
                }
            }
        }

    def generate_schemamap(self, cause: str, fields: List[Tuple[str, str]]) -> Dict[str, Any]:
        """生成schemamap结构"""
        return {
            "type": "object",
            "title": f"{cause}要素表",
            "properties": {
                "basicInfo": {
                    "type": "object",
                    "title": "文书基本信息",
                    "properties": {
                        "caseCode": {"type": "string", "title": "案号"},
                        "caseDesc": {"type": "string", "title": "案由"},
                        "trialCourt": {"type": "string", "title": "审理法院"},
                        "trialProcedure": {"type": "string", "title": "审判程序"},
                        "judgmentDate": {"type": "string", "title": "判决日期"}
                    }
                },
                "elementInfo": {
                    "type": "object",
                    "title": f"{cause}要素信息",
                    "properties": {
                        self.to_camel_case(field): {
                            "type": "string",
                            "title": desc.strip(),
                            "mapping": [field]
                        } for field, desc in fields
                    }
                }
            }
        }

    def save_to_jsonl(self, output_jsonl_path: str) -> None:
        """将结果保存为JSONL文件"""
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for cause, fields in self.case_fields.items():
                schema = self.generate_schema(cause, fields)
                schemamap = self.generate_schemamap(cause, fields)
                f.write(json.dumps(schema, ensure_ascii=False) + "\n")
                f.write(json.dumps(schemamap, ensure_ascii=False) + "\n")
        print(f"✅ JSONL 已写入：{output_jsonl_path}")


def extract_schema_to_jsonl(pdf_path: str, output_jsonl_path: str) -> None:
    """主函数"""
    extractor = SchemaExtractor()
    extractor.extract_fields(pdf_path)
    extractor.save_to_jsonl(output_jsonl_path)


# 示例用法
if __name__ == "__main__":
    pdf_file_path = "../documents/数据项列表-案由-06-06.pdf"
    output_jsonl_path = "../documents/output_schema.jsonl"
    extract_schema_to_jsonl(pdf_file_path, output_jsonl_path)
