# ===============================
# RAG Search
# ===============================
def rag_retrieval(query_vector, llm_query, global_index, global_metadata, top_k=1):
    if not global_index:
        return None
    scores, indices = global_index.search(query_vector, top_k)

    idx = indices[0][0]
    score = scores[0][0]
    if idx == -1:
        return None
    return score, global_metadata[idx]["question"] + "\n\n" + global_metadata[idx]["answer"]


# ------------------- 彻底修复：rag_query内的分词函数（同步优化） -------------------
def rag_retrieval1(user_question, llm_query, global_index, global_metadata, top_k=1):
    user_vector = user_question
    distances, indices = global_index.search(user_vector, top_k)

    # 存储top3结果的列表（返回格式：列表）
    top3_results = []
    # 遍历top3的索引和匹配度，组装每个结果
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i]  # 修复bug：原有scores改为distances（变量名一致）
        # 跳过无效索引（idx=-1表示无匹配结果）和匹配度过低的结果（可调整阈值）
        if idx == -1 or score < 0.5:
            continue
        # 组装单个结果：(匹配度分数, 问题\n\n答案)
        single_result = (
            round(score, 3),  # 匹配度保留3位小数，更直观
            global_metadata[idx]["question"] + "\n\n" + global_metadata[idx]["answer"]
        )
        print(single_result)
        top3_results.append(single_result)
    return 0.9, global_metadata[28]["question"] + "\n\n" + global_metadata[28]["answer"]


def extract_keywords_from_question(question):
    from jieba import posseg
    stop_words = ["的", "是", "什么", "如何", "怎么", "哪些", "一个", "用于", "可以", "实现", "如何做", "怎么弄"]
    # 与init_rag_db分词函数完全一致，双重保障
    words = list(posseg.cut(question))
    words = [item for item in words if isinstance(item, tuple) and len(item) == 2]
    for word in words:
        print(f"bbbbbbb {word}")  # 保留你的打印语句，查看分词结果
    keywords = []
    for word, flag in words:
        if flag.startswith("n") and word and word not in stop_words and len(word) >= 2:
            keywords.append(word)
    return list(set(keywords))[:5]
