import inspect
from typing import Type, Any, Union, ForwardRef
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from collections import abc


class LegalDocumentExtractor:
    """法律文书结构化信息提取器"""

    def build_enhanced_prompt(self, model_cls: Type[BaseModel], parser: BaseOutputParser) -> PromptTemplate:
        """
        构建包含完整约束说明的提示模板
        确保所有字段（包括多级嵌套）都显示正确的类型约束
        """

        def build_field_entry(name: str, field_info, indent=0, in_list=False) -> str:
            """递归构建字段描述条目"""
            prefix = "  " * indent + ("↳ " if in_list else "- ")
            type_name = self._get_type_name(field_info.annotation)
            description = field_info.description or "无描述"

            # 基础字段信息
            entry = [
                f"{prefix}{name}: {description}",
                f"{'  ' * (indent + 1)}• 类型: {type_name}",
                f"{'  ' * (indent + 1)}• 约束: {self._get_full_constraints(field_info)}"
            ]

            # 处理嵌套模型
            if self._is_pydantic_model(field_info.annotation):
                for sub_name, sub_field in field_info.annotation.model_fields.items():
                    entry.append(build_field_entry(sub_name, sub_field, indent + 1))

            # 处理容器类型
            elif self._is_generic_container(field_info.annotation):
                item_type = field_info.annotation.__args__[0] if field_info.annotation.__args__ else Any

                # 处理List[Model]
                if self._is_pydantic_model(item_type):
                    for sub_name, sub_field in item_type.model_fields.items():
                        entry.append(build_field_entry(sub_name, sub_field, indent + 1, True))

                # 处理List[List[...]]等嵌套容器
                elif self._is_generic_container(item_type):
                    entry.append(f"{'  ' * (indent + 1)}• 元素约束: {self._get_full_constraints(item_type)}")

                    # 添加最内层类型约束
                    inner_type = item_type.__args__[0] if item_type.__args__ else Any
                    if inner_type in (str, int, float, bool):
                        entry.append(
                            f"{'  ' * (indent + 1)}• 元素类型约束: {self._get_basic_type_constraints(inner_type)}"
                        )

            return "\n".join(entry)

        # 构建完整字段描述
        field_descriptions = []
        for field_name, field_info in model_cls.model_fields.items():
            field_descriptions.append(build_field_entry(field_name, field_info))

        return PromptTemplate(
            template="""你是一个法律裁判文书助手，负责从给定的文本中提取结构化法律信息。请严格遵守以下规则：
            
1. **内容来源限制**  
       - 只能提取文本中**明确出现**的信息，禁止添加、推测或虚构任何内容。  
       - 如果文本中没有相关字段信息，必须使用规定的空值格式填充（见下文）。
2. **字段规范与空值规则**  
       -见**字段规范说明**
3. **输出格式**  
       - 输出必须是一个完整的 JSON 对象，且严格符合给定的**字段规范说明**。  
       - 不能缺少任何字段。  
       - 字段名称和类型必须完全匹配规范。  
4. **禁止行为**  
       - 不允许增加任何额外字段或备注。  
       - 不允许省略字段。  
       - 不允许对文本内容进行解释或扩展。  

**字段规范说明（↳表示列表中的元素字段）**
{field_descriptions}

**待处理文本**
{input}

**必须严格遵守的输出格式**
{format_instructions}""",
            input_variables=["input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "field_descriptions": "\n\n".join(field_descriptions)
            }
        )

    # ========== 类型系统辅助方法 ==========
    def _get_type_name(self, type_hint) -> str:
        """获取可读的类型名称"""
        if hasattr(type_hint, "__origin__"):
            if type_hint.__origin__ is list or type_hint.__origin__ is abc.Sequence:
                return f"List[{self._get_type_name(type_hint.__args__[0])}]"
            elif type_hint.__origin__ is dict:
                return f"Dict[{', '.join(self._get_type_name(a) for a in type_hint.__args__)}]"
            elif type_hint.__origin__ is Union:
                return " | ".join(self._get_type_name(a) for a in type_hint.__args__)
        return getattr(type_hint, "__name__", str(type_hint))

    def _is_pydantic_model(self, obj) -> bool:
        """判断是否为Pydantic模型类"""
        try:
            if isinstance(obj, (ForwardRef, str)):
                # 处理前向引用
                return False

            if hasattr(obj, "model_fields"):
                return True

            if inspect.isclass(obj) and issubclass(obj, BaseModel):
                return True

            return False
        except Exception:
            return False

    def _is_generic_container(self, annotation) -> bool:
        """判断是否为泛型容器类型（List/Dict等）"""
        origin = getattr(annotation, "__origin__", None)
        return origin in (list, abc.Sequence, dict)

    # ========== 约束生成方法 ==========
    def _get_full_constraints(self, field_info) -> str:
        """生成完整的字段约束说明"""
        type_hint = field_info.annotation if hasattr(field_info, 'annotation') else field_info
        constraints = []

        # 判断是否 Optional[X]
        origin = getattr(type_hint, "__origin__", None)
        args = getattr(type_hint, "__args__", ())
        is_optional = origin is Union and type(None) in args

        # 获取真实类型（Optional[X] 的 X）
        base_type = None
        if is_optional:
            base_type = [a for a in args if a is not type(None)][0]
        else:
            base_type = type_hint

        # 基础类型约束
        if base_type == str:
            constraints.append("必须输出字符串")
            constraints.append("空值用null表示")
        elif base_type == int:
            constraints.append("必须输出整数")
            constraints.append("禁止引号包裹")
        elif base_type == float:
            constraints.append("必须输出浮点数")
            constraints.append("禁止引号包裹")
        elif base_type == bool:
            constraints.append("必须输出true/false")
            constraints.append("禁止使用是/否")

        # 容器类型约束
        elif self._is_generic_container(base_type):
            constraints.append("必须用[]表示数组" if base_type.__origin__ is list else "必须用{}表示字典")
            constraints.append("禁止用null表示空容器")

            if base_type.__args__:
                item_type = base_type.__args__[0]
                if item_type == str:
                    constraints.append("元素必须是字符串")
                elif item_type == int:
                    constraints.append("元素必须是整数")

        # Optional 补充说明
        if is_optional:
            constraints.append("允许为null")

        return "；".join(constraints) if constraints else "无特殊约束"

    def _get_full_constraints_bak(self, field_info) -> str:
        """生成完整的字段约束说明"""
        type_hint = field_info.annotation if hasattr(field_info, 'annotation') else field_info
        constraints = []

        # 基础类型约束
        if type_hint == str:
            constraints.extend(["必须输出字符串", "空值用null表示"])
        elif type_hint == int:
            constraints.extend(["必须输出整数", "禁止引号包裹"])
        elif type_hint == float:
            constraints.extend(["必须输出浮点数", "禁止引号包裹"])
        elif type_hint == bool:
            constraints.extend(["必须输出true/false", "禁止使用是/否"])

        # 容器类型约束
        elif self._is_generic_container(type_hint):
            constraints.append("必须用[]表示数组" if type_hint.__origin__ is list else "必须用{}表示字典")
            constraints.append("禁止用null表示空容器")

            # 元素类型约束
            if type_hint.__args__:
                item_type = type_hint.__args__[0]
                if item_type == str:
                    constraints.append("元素必须是字符串")
                elif item_type == int:
                    constraints.append("元素必须是整数")

        # 处理Optional[Type]
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union and type(None) in type_hint.__args__:
            constraints.append("允许为null")

        return "；".join(constraints) if constraints else "无特殊约束"

    def _get_basic_type_constraints(self, type_hint) -> str:
        """获取基础类型的约束说明"""
        constraints = {
            str: "必须输出字符串；空值用null表示",
            int: "必须输出整数；禁止引号包裹",
            float: "必须输出浮点数；禁止引号包裹",
            bool: "必须输出true/false",
            list: "必须用[]表示数组；禁止用null表示空数组",
            dict: "必须用{}表示字典；禁止用null表示空字典"
        }
        return constraints.get(type_hint, "")
