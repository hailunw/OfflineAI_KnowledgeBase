# ------------------- 核心改进：仅对question做分词提取关键字 -------------------
# 仅从question中提取核心关键字，不关联answer，简化逻辑
from jieba import posseg


def extract_keywords_from_question(question):
    """仅对question分词，提取核心关键字（过滤无意义词汇）"""

    # 1. 分词（优先提取名词、专业术语，过滤疑问词、虚词）
    words = posseg.cut(question)
    keywords = []
    # 过滤无意义词（可根据业务场景补充）
    stop_words = ["的", "是", "什么", "如何", "怎么", "哪些", "一个", "用于", "可以", "实现", "如何做", "怎么弄"]
    # 2. 筛选核心词汇：名词（n开头）、专业术语，长度≥2，非停用词
    for word, flag in words:
        if flag.startswith("n") and word not in stop_words and len(word) >= 2:
            keywords.append(word)
    # 3. 去重，避免冗余，取前5个核心关键字（足够支撑匹配，不占用过多资源）
    return list(set(keywords))[:5]
