标题: Self Introduction
内容: 
    Good morning. My name is Hailun Wang.
    
    I graduated in 2008 with a bachelor’s degree in Software Engineering (Japanese Strength), and I passed CET-6 during my university studies.
    
    After graduation, I worked at multinational companies like Infosys, where I gained solid experience in data warehousing, business intelligence, 
and backend development using Python and Java. I also achieved a score of 775 on the TOEIC test.

    Since moving to Japan, I have been engaged in a number of AI-related projects. For example, in the Offline AI Knowledge Base project, 
I used SentenceTransformer to generate vector embeddings from knowledge base documents, stored them in ChromaDB, and integrated large language models to 
build a natural-language conversational system for interactive knowledge base querying.

    Thanks for taking the time to interview me today. I really appreciate the chance to talk with you.

标题: AI离线AI知识库框架
内容: 
AI离线知识库系统采用 RAG（Retrieval-Augmented Generation）架构，
实现本地知识检索与本地大语言模型推理，系统可以完全离线运行。

系统整体流程如下：
   User Question
       ↓
   Query Rewrite（问题改写模块）
       ↓
   Embedding（文本向量化模块）
       ↓
   FAISS Search（向量检索模块）
       ↓
   RAG Retrieval（知识召回模块）
       ↓
   LLM Inference（本地LLM推理模块）
       ↓
   Answer（生成最终回答）
   
   系统主要由以下模块组成：   
   1 Markdown知识库模块：   负责读取本地 Markdown 文档作为知识库数据源。   
   2 Markdown解析模块：   将 Markdown 文档解析为 Question / Answer 结构。   
   3 Embedding向量化模块：   使用 SentenceTransformer 将文本转换为向量表示。   
   4 FAISS向量数据库模块：   使用 FAISS 存储向量并进行相似度检索。   
   5 Query Rewrite模块：   使用本地LLM对用户问题进行改写，生成更标准的问题。   
   6 RAG检索模块：   根据用户问题向量在FAISS中检索最相似的知识。   
   7 LLM推理模块：   如果检索到相关知识，则结合知识生成回答。   
   8 对话系统模块：   负责接收用户输入并输出最终回答。
   
   该系统特点：   
   - 完全离线运行
     - 支持Markdown知识库
     - 支持本地LLM推理
     - 支持向量检索
     - 支持问句改写
   - 


标题: 软件开发流程（AI离线知识库开发为例子）
内容: 标题: 异步，锁，多线程，多进程
内容: 


