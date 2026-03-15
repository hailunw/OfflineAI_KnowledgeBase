# ===============================
# Prompt
# ===============================

def build_query_prompt(input_query):
    """
    将非疑问句/词语转为"A是什么？"格式的标准疑问句
    :param input_query: 用户原始输入（如"自我介绍"）
    :return: 符合Qwen格式的prompt
    """
    # 移除多余缩进（避免Token浪费+格式干扰），规则表述更精准
    prompt = f"""<|im_start|>system
你是中文问句优化专家，严格按规则转换输入为标准疑问句。<|im_end|>
<|im_start|>user
请将"{input_query}"转换成标准疑问句，规则：
1. 必须保留"{input_query}"核心内容；
2. 以"？"结尾，格式为"A是什么？"；
3. 仅返回转换后的问句，无任何额外解释/内容。<|im_end|>
<|im_start|>assistant
"""
    return prompt


def build_answer_prompt(input_query):
    """
    生成指定规则的答案（20字内、常识、有意义）
    :param input_query: 标准疑问句（如"自我介绍是什么？"）
    :return: 符合Qwen格式的prompt
    """
    prompt = f"""<|im_start|>system
你是知识专家，回答严格遵守字数，内容和格式要求。<|im_end|>
<|im_start|>user
针对问句：{input_query}
请根据常识给出答案，规则：
1. 长度≤20个字，句号结尾的陈述句；
2. 答案有意义、符合常识；
3. 仅返回答案，无多余内容。
<|im_end|>
<|im_start|>assistant
"""
    return prompt
