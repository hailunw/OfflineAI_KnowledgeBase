import numpy as np


def extract_keywords_from_question(question):
    import jieba
    from jieba import posseg
    stop_words = ["的", "是", "什么", "如何", "怎么", "哪些", "一个", "用于", "可以", "实现", "如何做", "怎么弄"]
    # 关键修复：转为列表，避免生成器迭代失效
    words = list(posseg.cut(question))
    print(f"bbbbbbb {words}")  # 保留你的打印语句，查看分词结果
    keywords = []
    for word, flag in words:
        if flag.startswith("n") and word not in stop_words and len(word) >= 2:
            keywords.append(word)
    return list(set(keywords))[:5]

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