import os

from utils.constants import score_threshold
from utils.init import init_rag_tool, init_rag_db, init_llm
from utils.llm import build_query_prompt, build_answer_prompt
from utils.rag_db import rag_retrieval
from utils.rag_tool import text_2_vector


# ===============================
# Conversation
# ===============================

def main_conversation(rag_tool, global_index, global_metadata, llm_model):
    print("\n🎉 知识库问答系统")
    while True:
        query = input("👤 ").strip()
        if query in ["exit", "quit", "退出"]:
            break
        prompt = build_query_prompt(query)
        llm_query = ""
        for token in llm_model.create_completion(
                prompt=prompt,
                max_tokens=32,
                stream=True
        ):
            llm_query += token["choices"][0]["text"]
        llm_query = llm_query.strip()
        print(f"🔧 问题: {llm_query}")
        vect = text_2_vector(llm_query, rag_tool)
        score, results = rag_retrieval(vect, global_index, global_metadata)
        print(score)

        if results and score > score_threshold:
            print(f"📚 知识库答案: {results}")
        else:
            prompt = build_answer_prompt(llm_query)
            for token in llm_model.create_completion(
                    prompt=prompt,
                    max_tokens=128,
                    stream=True
            ):
                print(token["choices"][0]["text"], end="", flush=True)
            print()


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    print("CPU:", os.cpu_count())
    rag_tool = init_rag_tool()
    global_index, global_metadata = init_rag_db()
    llm_model = init_llm()
    main_conversation(rag_tool, global_index, global_metadata, llm_model)
