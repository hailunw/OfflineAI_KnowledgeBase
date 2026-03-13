import hashlib
import json
import os
import re

from utils.constants import FILE_INDEX_PATH, DOCS_DIR


# ===============================
# Markdown Splitter
# ===============================
class MarkdownSplitter:

    def split(self, text):
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        pattern = re.compile(r'标题:\s*(.*?)\n内容:\s*(.*?)(?=\n标题:|\Z)',re.S)

        matches = pattern.findall(text)

        results = []

        for q, a in matches:

            results.append({
                "question": q.strip(),
                "answer": a.strip()
            })

        return results


    def split_documents(self, documents):
        all_chunks = []
        for doc in documents:

            entries = self.split(doc.page_content)
            for i, entry in enumerate(entries):
                all_chunks.append({
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source','unknown')}_{i}"
                    }
                })
        print(f"📌 文档拆分完成：{len(all_chunks)} QA pairs")
        return all_chunks

# ===============================
# MD5
# ===============================

def calc_file_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def detect_changed_files():
    old_index = {}
    if os.path.exists(FILE_INDEX_PATH):
        with open(FILE_INDEX_PATH, "r", encoding="utf-8") as f:
            old_index = json.load(f)
    new_index = {}
    changed_files = []
    for md_file in DOCS_DIR.rglob("*.md"):
        path = str(md_file)
        md5 = calc_file_md5(md_file)
        new_index[path] = md5
        if path not in old_index:
            changed_files.append(path)
        elif old_index[path] != md5:
            changed_files.append(path)
    return changed_files, new_index
