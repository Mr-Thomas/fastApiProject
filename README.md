```
├── .env.dev                      # 开发环境的配置文件，包含 API 密钥等环境变量
├── .git/                         # Git 版本控制的元数据目录
├── .gitignore                    # Git 忽略文件，指定不需要纳入版本控制的文件和目录
├── .idea/                        # IntelliJ IDEA 等 IDE 的配置目录
├── app/                          # 项目核心应用代码目录
│   ├── api/                      # API 路由和控制器相关代码目录
│   │   ├── llm/                  # 大语言模型相关 API 代码目录
│   │   │   ├── file_controller.py # 文件处理相关 API 控制器
│   │   │   └── llm_controller.py  # 大语言模型调用相关 API 控制器
│   ├── core/                     # 项目核心配置和工具代码目录
│   │   ├── config.py             # 项目配置文件，定义应用设置和环境变量
│   │   ├── exception_handler.py  # 异常处理相关代码
│   │   └── exceptions.py         # 自定义异常类定义文件
│   ├── llm/                      # 大语言模型相关代码目录
│   │   ├── ollamaChatLLM.py      # Ollama 大语言模型封装代码
│   │   ├── tongyiLLM.py          # 通义千文大语言模型封装代码
│   │   └── zhipuLLM.py           # 智谱大语言模型封装代码
│   ├── main.py                   # FastAPI 应用入口文件，注册路由和异常处理器
│   ├── models/                   # 模型相关代码和文件目录
│   │   ├── bge-small-zh/         # BGE 小尺寸中文嵌入模型相关文件目录
│   │   │   ├── 1_Pooling/        # 池化层配置相关目录
│   │   │   │   └── config.json   # 池化层配置文件
│   │   │   ├── config.json       # 模型配置文件
│   │   │   ├── config_sentence_transformers.json # Sentence Transformers 相关配置文件
│   │   │   ├── modules.json      # 模型模块配置文件
│   │   │   ├── README.md         # 模型说明文档
│   │   │   ├── sentence_bert_config.json # Sentence BERT 相关配置文件
│   │   │   ├── special_tokens_map.json # 特殊标记映射配置文件
│   │   │   └── tokenizer_config.json # 分词器配置文件
│   │   └── init_bge.py           # 初始化 BGE 模型的脚本
│   ├── schemas/                  # Pydantic 数据模型定义目录
│   │   ├── base_response.py      # 基础响应数据模型定义文件
│   │   └── llm_schemas.py        # 大语言模型相关数据模型定义文件
│   ├── services/                 # 业务服务逻辑代码目录
│   │   ├── __init__.py           # 初始化服务模块，注册大语言模型服务
│   │   ├── file_service.py       # 文件处理服务代码
│   │   ├── llm_registry.py       # 大语言模型服务注册相关代码
│   │   ├── llm_service_factory.py # 大语言模型服务工厂类代码
│   │   ├── ollama_service.py     # Ollama 大语言模型服务代码
│   │   ├── tongyi_service.py     # 通义千文大语言模型服务代码
│   │   └── zhipuai_service.py    # 智谱大语言模型服务代码
│   ├── splitters/                # 文本分割和提取相关代码目录
│   │   └── text_splitter.py      # 文本分割和关键词提取相关代码
│   └── utils/                    # 工具函数和类代码目录
│       ├── clean_llm_output.py   # 清理大语言模型输出的工具函数
│       ├── document_process.py   # 文档处理相关工具函数和类
│       └── ocr_util.py           # OCR 识别相关工具函数
├── requirements.txt              # 项目依赖包列表文件
├── run.py                        # 运行 FastAPI 应用的脚本
├── test_main.http                # 测试 FastAPI 接口的 HTTP 请求文件
└── tests/                        # 测试代码目录
    ├── .pytest_cache/            # pytest 测试缓存目录
    └── test_hello.py             # 测试示例文件
```
