import json
import logging as py_logging
import os
import re
import warnings
from time import perf_counter

from transformers import logging

# =========================
# 全局日志/警告抑制（核心修复）
# =========================
# 关闭HuggingFace相关警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"  # 关闭llama_cpp日志
os.environ["FAISS_LOG_LEVEL"] = "2"  # 关闭faiss日志

# 关闭python内置警告
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 关闭transformers日志
logging.set_verbosity_error()
logging.disable_progress_bar()

# 关闭jieba日志
jieba_logger = py_logging.getLogger('jieba')
jieba_logger.setLevel(py_logging.ERROR)

# 关闭faiss日志
faiss_logger = py_logging.getLogger('faiss')
faiss_logger.setLevel(py_logging.ERROR)

# 关闭torch冗余日志
torch_logger = py_logging.getLogger('torch')
torch_logger.setLevel(py_logging.ERROR)

# 基础库导入（放在日志配置后）
import faiss
import jieba
import numpy as np
import torch
from llama_cpp import Llama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# =========================
# 常量定义（保持原格式）
# =========================
DOCS_DIR = "docs"
INDEX_PATH = "faiss_rag/index.bin"
META_PATH = "faiss_rag/metadata.json"
score_threshold = 0.65
llm_model_path = "llm_qwen/qwen2.5-3b-instruct-q4_k_m.gguf"
embedding_model = "sentence-transformers/paraphrase-MiniLM"


# =========================
# GPU检测（移除冗余打印）
# =========================
def check_gpu():
    try:
        if torch.cuda.is_available():
            print("GPU Mode")
            return True
        return False
    except Exception as err_msg:
        return False


# =========================
# 向量模型（保持原逻辑）
# =========================
def init_embedding_tool():
    if check_gpu():
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer(
        embedding_model,
        device=device,
        cache_folder="./cache"  # 增加缓存目录，避免重复下载
    )
    return model


# =========================
# LLM（保持原逻辑，移除冗余日志）
# =========================
def init_llm():
    use_gpu = check_gpu()
    # 禁用llama_cpp的控制台输出
    llm_kwargs = {
        "model_path": llm_model_path,
        "n_ctx": 128,
        "n_threads": 2,
        "verbose": False,  # 核心：关闭llama_cpp的详细日志
        "logits_all": False,
        "seed": 42
    }

    if use_gpu:
        llm_kwargs["n_gpu_layers"] = 6
        llm_kwargs["temperature"] = 0
    else:
        llm_kwargs["n_gpu_layers"] = 0

    llm = Llama(**llm_kwargs)
    return llm


# =========================
# Markdown解析（保持原逻辑）
# =========================
pattern = r'标题:\s*(.*?)\n内容:\s*(.*?)(?=\n标题:|\Z)'


def load_markdown():
    docs = []
    if not os.path.exists(DOCS_DIR):
        return docs

    for file in os.listdir(DOCS_DIR):
        if not file.endswith(".md"):
            continue
        path = os.path.join(DOCS_DIR, file)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            matches = re.findall(pattern, text, re.S)
            for q, a in matches:
                q_strip = q.strip()
                a_strip = a.strip()
                if q_strip and a_strip:  # 过滤空内容
                    docs.append({
                        "question": q_strip,
                        "answer": a_strip
                    })
        except Exception as e:
            # 静默处理文件读取错误，不打印冗余日志
            pass
    return docs


# =========================
# 三语言关键词（保持原逻辑）
# =========================
def generate_keywords(text):
    try:
        words = list(jieba.cut(text))
        keywords = []
        for w in words:
            if len(w) >= 2:
                keywords.append(w)
        keywords = list(set(keywords))[:3]
        return keywords
    except Exception:
        return []


# =========================
# 建立向量数据库（移除冗余打印）
# =========================
def build_rag_db(embed_model):
    docs = load_markdown()
    if not docs:
        return None, []

    questions = [d["question"] for d in docs]
    vectors = embed_model.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=False  # 关闭进度条
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = []
    for doc in docs:
        keywords = generate_keywords(doc["question"])
        metadata.append({
            "question": doc["question"],
            "answer": doc["answer"],
            "keywords": keywords
        })

    os.makedirs("faiss_rag", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return index, metadata


# =========================
# BM25（保持原逻辑）
# =========================
def build_bm25(metadata):
    corpus = []
    for m in metadata:
        tokens = jieba.lcut(m["question"])
        corpus.append(tokens)
    bm25 = BM25Okapi(corpus)
    return bm25


# =========================
# RAG检索（保持原逻辑）
# =========================
def rag_search(query, embed_model, index, metadata, bm25):
    if not query or not metadata or not bm25 or index is None:
        return None

    # ========== 第一步：使用faiss index做向量检索（修复数组处理） ==========
    vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # 用faiss index检索前10个候选
    scores, ids = index.search(vec, 10)

    # 关键修复1：将NumPy数组转为普通列表，避免歧义
    scores = scores[0].tolist()  # 转为列表
    ids = ids[0].tolist()  # 转为列表

    candidates = []
    candidate_scores = []  # 保存向量相似度分数（普通列表）
    for score, i in zip(scores, ids):
        # 关键修复2：增加i的有效性判断（避免负数索引）
        if i < 0 or i >= len(metadata) or score < score_threshold:
            continue
        candidates.append(metadata[i])
        candidate_scores.append(float(score))  # 确保是普通浮点数

    # 关键修复3：空候选直接返回
    if not candidates or len(candidate_scores) == 0:
        return None

    # ========== 第二步：BM25重排序（修复空值/数组问题） ==========
    # 1. 对向量候选集做BM25打分
    tokenized_query = jieba.lcut(query)
    candidate_questions = [c["question"] for c in candidates]
    candidate_corpus = [jieba.lcut(q) for q in candidate_questions]

    # 关键修复4：空语料库判断
    if not candidate_corpus or all(len(c) == 0 for c in candidate_corpus):
        return candidates[0]  # 退化为取第一个候选

    bm25_local = BM25Okapi(candidate_corpus)
    bm25_scores = bm25_local.get_scores(tokenized_query)

    # 关键修复5：转为普通列表+空值处理
    bm25_scores = [float(s) for s in bm25_scores]  # 转为普通列表
    if len(bm25_scores) == 0:
        return candidates[0]

    # 2. 融合向量分数和BM25分数（修复归一化除以0）
    # 关键修复6：避免除以0（加极小值）
    max_vec_score = max(candidate_scores) if max(candidate_scores) > 0 else 1e-6
    max_bm25_score = max(bm25_scores) if max(bm25_scores) > 0 else 1e-6

    normalized_vec = [s / max_vec_score for s in candidate_scores]
    normalized_bm25 = [s / max_bm25_score for s in bm25_scores]

    # 确保长度一致（防御性编程）
    min_len = min(len(normalized_vec), len(normalized_bm25))
    normalized_vec = normalized_vec[:min_len]
    normalized_bm25 = normalized_bm25[:min_len]

    combined_scores = [0.7 * v + 0.3 * b for v, b in zip(normalized_vec, normalized_bm25)]

    # ========== 第三步：选择最优结果（修复空数组） ==========
    if not combined_scores:
        return candidates[0]

    best_idx = np.argmax(combined_scores)
    # 关键修复7：索引越界保护
    best_idx = min(best_idx, len(candidates) - 1)

    return candidates[best_idx]


# =========================
# 问题改写（保持原逻辑，关闭冗余输出）
# =========================
def rewrite_question(llm, q):
    # 替换为中文Prompt，输出更符合预期
    prompt = f"""请将以下问题改写得更清晰、准确，便于检索：
    {q}
    改写后的问题："""
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=32,
            stop=["\n"],  # 防止输出过多
            echo=False  # 不回显prompt
        )
        return out["choices"][0]["text"].strip()
    except Exception:
        return q  # 失败时返回原问题


# =========================
# 主循环（仅保留核心交互输出）
# =========================
def main_conversation():
    # 初始化核心组件（保留你原有的初始化逻辑）
    embed_model = init_embedding_tool()

    # 加载/构建索引
    index = None
    metadata = []
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            index, metadata = build_rag_db(embed_model)
    else:
        index, metadata = build_rag_db(embed_model)

    # 初始化BM25和LLM
    bm25 = build_bm25(metadata) if metadata else None
    llm = init_llm()

    # 仅打印一次就绪提示
    print("RAG Ready")

    # 交互循环
    while True:
        try:
            q = input("👤 ")
            q_strip = q.strip()

            # 退出逻辑
            if q_strip == "exit":
                break
            # 空输入跳过
            if not q_strip:
                continue

            # ========== 修复点1：将计时放在所有逻辑之前 ==========
            start_time = perf_counter()
            estimated_time = 0  # 初始化计时变量，避免未定义

            # 问题优化
            q2 = rewrite_question(llm, q_strip)
            print("优化后的问题是:", q2)

            # 检索答案
            result = None
            if index and metadata and bm25:
                result = rag_search(q2, embed_model, index, metadata, bm25)

            # 输出结果
            if result:
                print("📚", result["answer"])
                # ========== 修复点2：在if分支也计算耗时 ==========
                end_time = perf_counter()
                estimated_time = end_time - start_time
            else:
                # LLM直接生成（保持原逻辑）
                out = llm.create_completion(
                    prompt=q_strip,
                    max_tokens=64,
                    stop=["\n"],
                    echo=False
                )
                response = out["choices"][0]["text"].strip()
                end_time = perf_counter()
                estimated_time = end_time - start_time
                print(response if response else "暂无相关答案")

            # ========== 修复点3：统一输出耗时（所有分支都能执行） ==========
            print(f"Estimated time: {estimated_time:.4f} seconds")  # 保留4位小数更美观

        except KeyboardInterrupt:
            print("\n程序退出")
            break
        except Exception as e:
            print(f"处理出错：{str(e)}，请重试")
            continue


if __name__ == "__main__":
    main_conversation()
