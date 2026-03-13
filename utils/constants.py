from pathlib import Path

# ===============================
# 全局变量
# ===============================

DOCS_DIR = Path(r"C:\Users\shuei\Desktop\OfflineAI_KnowledgeBase\docs")
llm_model_path = r"C:\Users\shuei\Desktop\OfflineAI_KnowledgeBase\llm_qwen\qwen2.5-1.5b-instruct.Q4_K_M.gguf"
sentence_transform_model = r"C:\Users\shuei\Desktop\OfflineAI_KnowledgeBase\paraphrase-MiniLM"
FAISS_DIR = "faiss_rag"
INDEX_PATH = f"{FAISS_DIR}/index.bin"
META_PATH = f"{FAISS_DIR}/metadata.json"
FILE_INDEX_PATH = f"{FAISS_DIR}/file_index.json"
