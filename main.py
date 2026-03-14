import os
from time import perf_counter

# 仅保留必要依赖，移除llama_cpp相关校验
import torch

from utils.constants import score_threshold
from utils.init import init_rag_tool, init_rag_db, init_llm
from utils.llm import build_query_prompt, build_answer_prompt
from utils.rag_db import rag_retrieval
from utils.rag_tool import text_2_vector


# ===============================
# GPU适配初始化（纯兼容性版本）
# ===============================
def check_gpu_availability():
    """校验MX150 GPU是否可用（无任何高版本依赖）"""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "无GPU"
    mx150_detected = cuda_available and "MX150" in gpu_name

    print(f"📌 GPU状态检测：")
    print(f"   - CUDA可用: {cuda_available}")
    print(f"   - 显卡型号: {gpu_name}")

    if mx150_detected:
        print(f"   - 适配MX150：显存限制2G，已在LLM初始化时配置GPU")
        return True
    elif cuda_available:
        print(f"   - 非MX150显卡，GPU已在初始化时启用")
        return True
    else:
        print(f"⚠️  未检测到可用GPU，使用CPU推理")
        return False


# ===============================
# Conversation（移除所有create_completion的GPU参数）
# ===============================
def main_conversation(rag_tool, global_index, global_metadata, llm_model):
    print("\n🎉 知识库问答系统")
    # 仅检测GPU状态，不传递参数到create_completion
    use_gpu = check_gpu_availability()

    while True:
        query = input("👤 ").strip()
        if query in ["exit", "quit", "退出"]:
            break
        start_time = perf_counter()

        # 1. LLM生成优化后的问题（仅保留核心参数，移除n_gpu_layers/n_threads）
        prompt = build_query_prompt(query)
        llm_query = ""
        for token in llm_model.create_completion(
                prompt=prompt,
                max_tokens=24,  # MX150显存优化：减少生成长度
                stream=True,
                temperature=0.0,  # 固定温度，减少计算量
                top_p=0.95
        ):
            llm_query += token["choices"][0]["text"]
        llm_query = llm_query.strip()
        print(f"🔧 问题: {llm_query}")

        # 2. 向量编码（GPU已在init_rag_tool中配置）
        vect_quest = text_2_vector(llm_query, rag_tool)
        # 修复参数错误：仅传递3个必要参数
        retrieval_result = rag_retrieval(vect_quest,llm_query, global_index, global_metadata)

        # 3. 处理top3返回结果
        score = 0.0
        results = None
        if retrieval_result and len(retrieval_result) > 0:
            score, results = retrieval_result

        # 4. 输出结果
        elapsed_time = perf_counter() - start_time
        print(f"可信度: {score:.2f}, 用时: {elapsed_time:.4f} 秒")

        if results and score > score_threshold:
            print(f"📚 知识库答案: {results}")
        else:
            # LLM生成答案（仅保留核心参数，无GPU相关参数）
            prompt = build_answer_prompt(llm_query)
            print("🤖 LLM答案: ", end="", flush=True)
            for token in llm_model.create_completion(
                    prompt=prompt,
                    max_tokens=48,  # MX150显存优化
                    stream=True,
                    temperature=0.0,
                    top_p=0.95
            ):
                print(token["choices"][0]["text"], end="", flush=True)
            print("\n")


# ===============================
# Main（保持初始化逻辑不变）
# ===============================
if __name__ == "__main__":
    print("CPU核心数:", os.cpu_count())

    # 1. 初始化RAG工具（强制调用GPU，device=cuda）
    rag_tool = init_rag_tool()
    if hasattr(rag_tool, 'device'):
        print(f"📌 向量模型设备: {rag_tool.device}")

    # 2. 初始化RAG数据库
    global_index, global_metadata = init_rag_db()

    # 3. 初始化LLM（GPU参数仅在初始化时配置，不在生成时传递）
    llm_model = init_llm()

    # 4. 启动对话
    main_conversation(rag_tool, global_index, global_metadata, llm_model)