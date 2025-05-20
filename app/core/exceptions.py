from typing import Any


class BizException(Exception):
    def __init__(self, code: int = 4001, message: str = "Business Error", data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")
