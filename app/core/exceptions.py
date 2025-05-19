class BizException(Exception):
    def __init__(self, code: int = 4001, message: str = "Business Error"):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
