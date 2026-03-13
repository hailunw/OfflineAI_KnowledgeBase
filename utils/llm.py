# ===============================
# Prompt
# ===============================

def build_query_prompt(input_query):
    prompt = f"""<|im_start|>system
    你是一个中文专家，擅长将非疑问句或者词语转换成“A是什么？”这样的标准疑问句。<|im_end|>
    <|im_start|>user
    请将"{input_query}"转换成标准疑问句？
    规则：①必须保留"{input_query}" ②加"？"结尾 ③不解释 ④不额外输出。
    <|im_end|>
    <|im_start|>assistant
    """

    return prompt


def build_answer_prompt(input_query):
    prompt = f"""<|im_start|>system
            你是一名知识渊博的知识库专家。<|im_end|>
            <|im_start|>user
            请针对问句{input_query}，根据常识给出长度在20个字之内的有意义的答案。
            <|im_end|>
            <|im_start|>assistant
            """

    return prompt
