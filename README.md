```
├── app/                          # 项目核心应用代码目录
│   ├── api/                      # API 路由和控制器相关代码目录
│   ├── core/                     # 项目核心配置和工具代码目录
│   ├── llm/                      # 大语言模型相关代码目录
│   ├── main.py                   # FastAPI 应用入口文件，注册路由和异常处理器
│   ├── local_models/             # 本地打模型相关
│   ├── schemas/                  # Pydantic 数据模型定义目录
│   ├── services/                 # 业务服务逻辑代码目录
│   ├── splitters/                # 文本分割和提取相关代码目录
│   └── utils/                    # 工具函数和类代码目录
├── requirements.txt              # 项目依赖包列表文件
├── run.py                        # 运行 FastAPI 应用的脚本
├── test_main.http                # 测试 FastAPI 接口的 HTTP 请求文件
└── tests/                        # 测试代码目录
```
