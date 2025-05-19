import re


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
