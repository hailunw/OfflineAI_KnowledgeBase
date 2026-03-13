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