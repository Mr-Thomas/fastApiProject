from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    model = SentenceTransformer("BAAI/bge-small-zh")
    model.save("./bge-small-zh")
