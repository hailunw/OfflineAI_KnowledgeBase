# ===============================
# RAG Tool
# ===============================
import json
import os

import faiss
import numpy as np
from langchain_community.document_loaders import TextLoader
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from utils.constants import llm_model_path, FILE_INDEX_PATH, FAISS_DIR, META_PATH, INDEX_PATH, rag_tool_model_path
from utils.markdown import MarkdownSplitter, detect_changed_files
from utils.rag_tool import text_2_vector

llm_model = None
global_index = None
global_metadata = []
rag_tool = None


def init_rag_tool():
    global rag_tool

    if rag_tool is None:
        rag_tool = SentenceTransformer(
            rag_tool_model_path,
            device="cpu",
            cache_folder="./cache"
        )

    return rag_tool


# ===============================
# RAG DB
# ===============================

def init_rag_db():

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        global_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            global_metadata = json.load(f)
        print(f"✅ 初始化RAG知识库，已加载FAISS索引: {global_index.ntotal}")
    else:
        global_index = None
        global_metadata = []
    changed_files, new_file_index = detect_changed_files()
    if not changed_files:
        print("📂 Markdown无变化")
        return global_index, global_metadata
    print(f"🆕 检测到 {len(changed_files)} 个文件变化")
    documents = []
    for file in changed_files:
        loader = TextLoader(file, encoding="utf-8")
        documents.extend(loader.load())
    splitter = MarkdownSplitter()
    qa_docs = splitter.split_documents(documents)

    if not qa_docs:
        print("⚠️ 未解析出QA")
        return global_index, global_metadata
    # 只向量化 question
    questions = [doc["question"] for doc in qa_docs]
    print("⚡ embedding question...")
    vectors = []
    for question in questions:
        vector = text_2_vector(question, rag_tool)
        print(f"question: {question}")
        print(f"vector shape: {vector.shape}")
        vectors.append(vector[0])  # 关键
    vectors = np.array(vectors).astype("float32")
    if global_index is None:
        vector_dim = vectors.shape[1]
        global_index = faiss.IndexFlatIP(vector_dim)

    global_index.add(vectors)
    # metadata 保存 question + answer
    global_metadata.extend(
        [
            {
                "question": doc["question"],
                "answer": doc["answer"],
                "metadata": doc["metadata"]
            }
            for doc in qa_docs
        ]
    )
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(global_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(global_metadata, f, ensure_ascii=False, indent=2)
    with open(FILE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(new_file_index, f, indent=2)
    print(f"✅ 新增 {len(vectors)} QA向量")
    return global_index, global_metadata


# ===============================
# LLM
# ===============================

def init_llm():
    print("🤖 加载 离线版LLM")

    # llm = Llama(
    #     model_path=llm_model_path,
    #     n_ctx=1024,
    #     n_threads=os.cpu_count(),
    #     n_gpu_layers=0,
    #     temperature=0,
    #     verbose=False
    # )
    llm = Llama(
        model_path=llm_model_path,  # 替换为实际路径
        n_ctx=512,  # CPU推荐256/512，GPU可设2048
        n_threads=os.cpu_count(),  # 拉满CPU核心
        n_gpu_layers=0,  # 纯CPU=0，有GPU填≥1
        temperature=0.0,  # 0=最快/确定，0.3=兼顾灵活
        verbose=False,  # 关闭冗余日志
        f16_kv=True,  # 提速+省内存（必开）
        use_mlock=True,  # 锁定模型到内存（纯CPU必开）
    )
    return llm
