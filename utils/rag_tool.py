# ===============================
# Embedding
# ===============================


def text_2_vector(text, rag_tool):
    vector = rag_tool.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    if isinstance(text, str):
        vector = vector.reshape(1, -1)

    return vector
