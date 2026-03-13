from pathlib import Path

# ===============================
# 全局变量
# ===============================

DOCS_DIR = Path("docs")
llm_model_path = "llm_qwen\qwen2.5-1.5b-instruct.Q4_K_M.gguf"
rag_tool_model_path = "paraphrase-MiniLM"
FAISS_DIR = "faiss_rag"
INDEX_PATH = f"{FAISS_DIR}/index.bin"
META_PATH = f"{FAISS_DIR}/metadata.json"
FILE_INDEX_PATH = f"{FAISS_DIR}/file_index.json"
