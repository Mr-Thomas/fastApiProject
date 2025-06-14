from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    """
    下载模型并保存
     https://github.com/FlagOpen/FlagEmbedding
     BAAI/bge-large-zh
     BAAI/bge-m3
     BAAI/bge-small-zh
     BAAI/bge-base-zh
    """
    model = SentenceTransformer("BAAI/bge-base-zh")
    model.save("./bge-base-zh")
