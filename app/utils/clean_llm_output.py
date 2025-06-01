import json
import re
from typing import Any
from markdown2 import markdown as md_to_html  # 引入 markdown 转 HTML 库  pip install markdown2
import html

from app.core.exceptions import BizException


def extract_response_content(response: Any) -> str:
    """
    自动判断响应类型（自然语言 or 函数调用），并提取内容。

    返回:
        - 如果是普通文本，则返回 content 或 text。
        - 如果是函数调用，则返回 function_call.arguments。
    """
    try:
        output = getattr(response, "output", None)
        if not output:
            raise BizException(code=500, message="LLM response missing 'output' field.")

        # 优先判断是否为结构化 choices 响应（Function Call or Chat Response）
        choices = getattr(output, "choices", None)
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = getattr(choices[0], "message", None)
            if not message:
                raise BizException(code=500, message="LLM response missing 'message' in choices.")

            # --- 如果是函数调用响应 ---
            if isinstance(message, dict):
                function_call = message.get("function_call")
            else:
                function_call = getattr(message, "function_call", None)
            if function_call:
                if isinstance(function_call, dict):
                    return function_call.get("arguments", "")
                elif hasattr(function_call, "arguments"):
                    return getattr(function_call, "arguments", "")
                else:
                    raise BizException(code=500, message="Malformed function_call, missing 'arguments'.")

            # --- 如果是自然语言响应 ---
            return getattr(message, "content", "")

        # --- 如果是普通 output.text 结构 ---
        return getattr(output, "text", "")

    except Exception as e:
        raise BizException(code=500, message=f"Failed to parse LLM response: {str(e)}")


def blocks_to_markdown_html(blocks):
    markdown_parts = []
    html_parts = []

    for block in blocks:
        block_type = block.get("type")

        if block_type == "title":
            text = block.get("text", "")
            markdown_parts.append(f"# {text}")
            html_parts.append(f"<h1>{html.escape(text)}</h1>")

        elif block_type == "paragraph":
            text = block.get("text", "")
            markdown_parts.append(text)
            html_parts.append(f"<p>{html.escape(text)}</p>")

        elif block_type == "list":
            items = block.get("items", [])
            markdown_parts.extend([f"- {item}" for item in items])
            html_list = "".join(f"<li>{html.escape(item)}</li>" for item in items)
            html_parts.append(f"<ul>{html_list}</ul>")

        elif block_type == "table":
            md_table = block.get("markdown", "")
            markdown_parts.append(md_table)
            html_parts.append(md_to_html(md_table))  # 直接转 HTML

        elif block_type == "image":
            src = block.get("src") or block.get("path", "")
            caption = block.get("caption", "")
            # Markdown 语法
            markdown_parts.append(f"![{caption}]({src})")
            # HTML 图文结构
            html_parts.append(
                f"<figure><img src='{html.escape(src)}' style='max-width:100%;'/><figcaption>{html.escape(caption)}</figcaption></figure>"
            )

    markdown_output = '\n\n'.join(markdown_parts)
    html_output = '\n'.join(html_parts)
    return markdown_output, html_output


def extract_json_block(text: str) -> str:
    """
    从文本中提取 Markdown 格式中的 JSON 代码块，支持 ```json 或 ```json 开始的代码块。
    """

    # 先找有结尾的代码块
    pattern_with_end = r"```json\s*\n([\s\S]*?)\n\s*```"
    match = re.search(pattern_with_end, text)
    if match:
        return match.group(1).strip()
    # 找没有结尾的代码块，从```json开始到文本末尾
    pattern_no_end = r"```json\s*\n([\s\S]*)"
    match = re.search(pattern_no_end, text)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_llm_output_robust(text: str, tag: str = "think", remove_tag_content: bool = True) -> str:
    """
    更健壮地清理 LLM 输出中的指定标签及非 JSON 内容，尝试提取纯 JSON 字符串。

    :param text: 原始 LLM 输出文本
    :param tag: 要清理的标签名，默认 "think"
    :param remove_tag_content: 是否连带标签内内容一起删除，默认 True
    :return: 清理后的文本，理论上是纯 JSON 字符串
    """
    # 1. 删除指定标签及其中内容
    if remove_tag_content:
        pattern = rf"<{tag}.*?>.*?</{tag}>"
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    else:
        # 只删除标签，保留内容
        pattern = rf"</?{tag}.*?>"
        text = re.sub(pattern, "", text)

    text = text.strip()

    # 2. 尝试抽取第一段 JSON 对象（{}）或数组（[]）
    # 找到第一个 '{' 或 '[' 作为起点
    start_pos = None
    for c in ['{', '[']:
        pos = text.find(c)
        if pos != -1 and (start_pos is None or pos < start_pos):
            start_pos = pos
    if start_pos is None:
        # 找不到 JSON 起始符号，直接返回原文本
        return text

    # 从起始位置开始尝试截取有效 JSON
    substr = text[start_pos:]
    # 利用栈方法匹配完整的 JSON 对象或数组
    stack = []
    end_pos = None
    for i, ch in enumerate(substr):
        if ch in ['{', '[']:
            stack.append(ch)
        elif ch in ['}', ']']:
            if not stack:
                # 不匹配，忽略
                break
            opening = stack.pop()
            if (opening == '{' and ch != '}') or (opening == '[' and ch != ']'):
                # 不匹配
                break
            if not stack:
                end_pos = i + 1
                break
    if end_pos is None:
        # 没有匹配到完整 JSON，返回清理后的全文
        return text

    json_str = substr[:end_pos]

    # 3. 最后尝试用 json.loads 校验一次，如果不合法则返回原文本
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        return text


def clean_llm_output(tag: str, text: str, remove_content: bool = False) -> str:
    """
    清理指定标签的内容
    :param tag: 要清除的标签名，例如 'think'
    :param text: 原始文本
    :param remove_content: 是否连带标签内内容也删除，默认只去标签
    :return: 清理后的文本
    """
    if remove_content:
        # 删除整个标签及内容
        pattern = rf"<{tag}.*?>.*?</{tag}>"
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()
    else:
        # 仅删除标签本身，保留内容
        start_tag = rf"</?{tag}.*?>"
        return re.sub(start_tag, "", text).strip()


if __name__ == '__main__':
    # 测试用例
    text = """
    ```json
        {
          "court_name": "未提供",
          "court_code": "未提供",
          "cause_of_action": "离婚纠纷",
          "filing_date": null,
          "trial_procedure": "未提供",
          "judgment_date": "2022-08-25",
          "judges": [
            {
              "role": "审判员",
              "name": "朱元勋"
            },
            {
              "role": "书记员",
              "name": "杨帅"
            }
          ],
          "parties": null,
          "claim_defense": null,
          "evidences": null,
          "fact_findings": null,
          "legal_judgment": {
            "legal_basis": [
              "《中华人民共和国民法典》第一千零四十三条",
              "《中华人民共和国民法典》第一千零七十九条"
            ],
            "judgment_result": "未提供"
          },
          "other_info": null
        }
        ```  
        富士达v撒v阿萨萨格打过去
    """
    tt = extract_json_block(text)
    print(tt)
