# ===============================
# RAG Search
# ===============================

def rag_retrieval(query_vector, global_index, global_metadata, top_k=1):
    if not global_index:
        return None
    scores, indices = global_index.search(query_vector, top_k)

    idx = indices[0][0]

    if idx == -1:
        return None
    return global_metadata[idx]["question"] + "\n\n" + global_metadata[idx]["answer"]
