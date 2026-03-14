# ===============================
# RAG Tool
# ===============================
import json
import os

import faiss
import numpy as np
import torch
from langchain_community.document_loaders import TextLoader
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from utils.constants import llm_model_path, FILE_INDEX_PATH, FAISS_DIR, META_PATH, INDEX_PATH, rag_tool_model_path, \
    gpu_n_threads, cpu_n_threads
from utils.markdown import MarkdownSplitter, detect_changed_files
from utils.rag_tool import text_2_vector

llm_model = None
global_index = None
global_metadata = []
rag_tool = None


def init_rag_tool():
    global rag_tool
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CPU Type: {device_type}")
    if rag_tool is None:
        rag_tool = SentenceTransformer(
            rag_tool_model_path,
            device=device_type,  # cuda cpu
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

    # ------------------- 彻底修复：init_rag_db内的分词函数（杜绝迭代报错） -------------------
    def extract_keywords_from_question(question):
        """仅对question分词，提取核心关键字（过滤无意义词汇）- 彻底修复迭代解包报错"""
        from jieba import posseg
        stop_words = ["的", "是", "什么", "如何", "怎么", "哪些", "一个", "用于", "可以", "实现", "如何做", "怎么弄"]
        # 新增：格式校验+转换，避免传入非字符串（如numpy数组）导致decode报错
        if not isinstance(question, str):
            # 若为numpy数组，转为字符串；其他类型直接转为字符串
            if isinstance(question, np.ndarray):
                question = question.astype(str).tolist()
                question = " ".join(question) if isinstance(question, list) else str(question)
            else:
                question = str(question)
        # 双重保障：1. 转为列表 2. 过滤异常值，确保每个元素都是可解包的（word, flag）对
        words = list(posseg.cut(question))
        # 过滤无效元素（避免非（word,flag）对导致的解包报错）
        words = [item for item in words if isinstance(item, tuple) and len(item) == 2]
        # 可选：打印分词结果，查看是否有异常（注释后不影响功能）
        # print(f"init分词结果: {words}")
        keywords = []
        for word, flag in words:
            # 额外校验：避免word/flag为空导致的异常
            if flag.startswith("n") and word and word not in stop_words and len(word) >= 2:
                keywords.append(word)
        return list(set(keywords))[:5]

    # ------------------------------------------------------------------

    # 向量化：question + 提取的关键字（逻辑不变，分词函数已彻底修复）
    print("⚡ embedding question + 核心关键字...")
    vectors = []
    for doc in qa_docs:
        question = doc["question"]
        # 调用彻底修复后的分词函数，杜绝迭代解包报错
        keywords = extract_keywords_from_question(question)
        query_text = f"{question} {' '.join(keywords)}"
        vector = text_2_vector(query_text, rag_tool)
        vectors.append(vector[0])

    vectors = np.array(vectors).astype("float32")
    if global_index is None:
        vector_dim = vectors.shape[1]
        global_index = faiss.IndexFlatIP(vector_dim)

    global_index.add(vectors)
    global_metadata.extend(
        [
            {
                "question": doc["question"],
                "answer": doc["answer"],
                "metadata": doc["metadata"],
                "keywords": extract_keywords_from_question(doc["question"])
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

    print(f"✅ 新增 {len(vectors)} QA向量（分词函数彻底修复，无迭代解包报错）")
    return global_index, global_metadata


# ===============================
# LLM
# ===============================

def init_llm():
    """初始化LLM（MX150 GPU适配，低版本llama-cpp兼容）"""
    # MX150 GPU检测
    use_gpu = torch.cuda.is_available()
    mx150_detected = False
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        mx150_detected = "MX150" in gpu_name

    if use_gpu:
        print(f"🚀 使用GPU推理（{gpu_name}）- MX150适配版")
        llm = Llama(
            model_path=llm_model_path,
            n_ctx=256,  # 解决n_ctx警告，适配MX150显存
            n_threads=gpu_n_threads,  # MX150搭配CPU最优线程数（仅初始化时配置）
            n_gpu_layers=12 if mx150_detected else 20,  # GPU层仅初始化时配置
            temperature=0.0,
            top_p=0.95,
            f16_kv=True,  # 启用半精度加速GPU
            verbose=False  # 关闭冗余日志
        )
    else:
        print("💻 未检测到GPU，使用CPU推理")
        llm = Llama(
            model_path=llm_model_path,
            n_ctx=256,
            n_threads=cpu_n_threads,
            n_gpu_layers=20,
            temperature=0.0,
            top_p=0.95,
            f16_kv=True,
            verbose=False
        )

    return llm
