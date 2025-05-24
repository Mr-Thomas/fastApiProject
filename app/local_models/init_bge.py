from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    # BAAI/bge-small-zh
    # model = SentenceTransformer("BAAI/bge-small-zh")
    # model.save("./bge-small-zh")

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
    # 保存模型和分词器
    model.save_pretrained("./Qwen2.5-VL-3B-Instruct")
    tokenizer.save_pretrained("./Qwen2.5-VL-3B-Instruct")
