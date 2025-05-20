import json
import re


def extract_json_block(text: str) -> str:
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
