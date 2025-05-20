from typing import Union

from langchain_core.prompts import PromptTemplate


class LegalDocumentExtractor:

    def _build_enhanced_prompt(self, model_cls, parser) -> PromptTemplate:
        """完整显示所有嵌套字段约束（含多级列表）的提示模板"""

        def build_field_entry(name: str, field, indent=0, is_list_item=False) -> str:
            prefix = "  " * indent + ("- " if not is_list_item else "↳ ")
            type_name = self._get_type_name(field.annotation)

            # 基础字段描述
            entry = f"{prefix}{name}: {field.description or '无描述'}\n" + \
                    f"{'  ' * (indent + 1)}• 类型: {type_name}\n" + \
                    f"{'  ' * (indent + 1)}• 约束: {self._get_constraints(field)}"

            # 处理嵌套模型
            if self._is_pydantic_model(field.annotation):
                for n, f in field.annotation.model_fields.items():
                    entry += "\n" + build_field_entry(n, f, indent + 1)

            # 处理List[Model]
            elif self._is_list_of_models(field.annotation):
                item_type = field.annotation.__args__[0]
                if self._is_pydantic_model(item_type):
                    for n, f in item_type.model_fields.items():
                        entry += "\n" + build_field_entry(n, f, indent + 1, is_list_item=True)

            # 处理List[List[...]]
            elif self._is_list_of_lists(field.annotation):
                entry += "\n" + f"{'  ' * (indent + 1)}• 元素约束: {self._get_constraints(field.annotation.__args__[0])}"

            return entry

        field_descriptions = []
        for name, field in model_cls.model_fields.items():
            field_descriptions.append(build_field_entry(name, field))

        return PromptTemplate(
            template="""作为法律文书结构化提取系统，请严格遵循以下要求：

                **字段规范（↳表示列表元素字段）**
                {field_descriptions}
            
                **全局规则**
                1. 严格按标注的类型和约束输出
                2. 字符串：空值用null | 数组：空值用[]
                3. 禁止任何解释性文字
                4. 时间格式：YYYY-MM-DD
            
                **输入文本**
                {input}
            
                **必须遵守的输出格式**
                {format_instructions}""",
            input_variables=["input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "field_descriptions": "\n\n".join(field_descriptions)
            }
        )

    # 新增辅助方法
    def _is_list_of_lists(self, annotation) -> bool:
        """判断是否是List[List[...]]类型"""
        return (hasattr(annotation, "__origin__")
                and annotation.__origin__ is list
                and len(annotation.__args__) == 1
                and hasattr(annotation.__args__[0], "__origin__")
                and annotation.__args__[0].__origin__ is list)

    # 辅助方法
    def _get_constraints(self, field) -> str:
        """生成字段约束说明"""
        type_hint = field.annotation
        constraints = []

        # 基础类型约束
        if type_hint == str:
            constraints.append("必须输出字符串")
            constraints.append("空值用null表示")
        elif type_hint == int:
            constraints.append("必须输出整数")
            constraints.append("不要引号包裹")
        elif type_hint == list:
            constraints.append("必须用[]表示数组")
            constraints.append("禁止用null表示空数组")

        # 处理Optional[Type]
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
            constraints.append("允许为null")

        return "；".join(constraints)

    def _is_list_of_models(self, annotation) -> bool:
        """判断是否是List[Model]类型"""
        return (hasattr(annotation, "__origin__")
                and annotation.__origin__ is list
                and len(annotation.__args__) == 1
                and self._is_pydantic_model(annotation.__args__[0]))

    # 辅助方法
    def _get_type_name(self, type_hint) -> str:
        """获取可读的类型名称"""
        if hasattr(type_hint, "__origin__"):
            if type_hint.__origin__ == list:
                return f"List[{self._get_type_name(type_hint.__args__[0])}]"
            elif type_hint.__origin__ == dict:
                return f"Dict[{', '.join(self._get_type_name(a) for a in type_hint.__args__)}]"
        return getattr(type_hint, "__name__", str(type_hint))

    def _is_pydantic_model(self, obj) -> bool:
        """判断是否为Pydantic模型"""
        return hasattr(obj, "model_fields")

    def _is_generic_container(self, obj) -> bool:
        """判断是否为泛型容器"""
        return hasattr(obj, "__origin__") and obj.__origin__ in (list, dict)
