import os
import re
from pathlib import Path

# ===============================
# 全局变量
# ===============================

DOCS_DIR = Path("docs")
# llm_model_path = r"llm_qwen\qwen2.5-1.5b-instruct.Q4_K_M.gguf"
llm_model_path = r"llm_qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
rag_tool_model_path = "paraphrase-MiniLM"
# rag_tool_model_path = "paraphrase-multilingual-MiniLM-L12-v2"

FAISS_DIR = "faiss_rag"
INDEX_PATH = f"{FAISS_DIR}/index.bin"
META_PATH = f"{FAISS_DIR}/metadata.json"
FILE_INDEX_PATH = f"{FAISS_DIR}/file_index.json"
score_threshold = 0.60

markdown_split_pattern = r'标题:\s*(.*?)\n内容:\s*(.*?)(?=\n标题:|\Z)'

cpu_n_threads = os.cpu_count(),
gpu_n_threads = 2

