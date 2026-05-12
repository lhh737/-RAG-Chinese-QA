<p align="center">
  <h1 align="center">RAG-Chinese-QA</h1>
  <p align="center">基于 LangChain 的中文 RAG 文档问答系统 — 混合检索 · HyDE 增强 · 重排序 · 答案溯源</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/LangChain-1.2.13-orange.svg" alt="LangChain">
</p>

---

## 概述

一个完整的 Retrieval-Augmented Generation (RAG) 系统，专为中文文档问答场景设计。上传 PDF/TXT 文档后，系统自动完成分块、向量化、索引构建，随后可通过自然语言提问获取带溯源的精确答案。

**核心理念**：取多种检索策略之长，通过"粗筛→精排→上下文还原"三级管线，在不显著增加延迟的前提下最大化召回率与答案质量。

## 检索管线

```
用户提问
  │
  ├─→ [HyDE] LLM 生成假设性答案作为检索 query
  │
  ├─→ [粗筛] FAISS 向量检索 ─┐
  │         BM25 稀疏检索  ─┤ 合并去重，候选池 40-60 条
  │                         ┘
  ├─→ [精排] Cross-Encoder / DashScope Reranker → Top 12
  │
  ├─→ [还原] 子块 → 父块映射，动态 TopK (4-8 块)
  │
  └─→ [生成] Qwen 结合上下文生成答案 + 溯源引用
```

## 核心特性

### 混合检索

FAISS 稠密向量 + BM25 稀疏检索双路并行召回。向量擅长语义匹配，BM25 擅长关键词命中，两者互补，显著降低漏召率。

- 向量路：DashScope `text-embedding-async-v2` / BGE-M3 嵌入，FAISS 索引
- 稀疏路：jieba 中文分词，BM25Okapi 倒排索引，精确匹配中文术语

### HyDE 增强

用户提问往往短且口语化，直接检索效果不佳。HyDE (*Hypothetical Document Embeddings*) 先让 LLM 将问题展开为一段"假想文档"，再用这段文档作为检索 query，大幅提升与知识库文本的语义对齐度，从而提升召回率。LLM 调用失败时自动回退为原始 query。

### 父子分块

| 类型 | 大小 | 用途 |
|------|------|------|
| 子块 (child) | ~400 字 | 精准检索，送入 FAISS + BM25 |
| 父块 (parent) | ~1200 字 | 上下文还原，保证语义完整 |

检索在子块上进行，返回时将关联的父块内容一并提供给 LLM，避免因分块切断关键上下文。

### 重排序

从候选池中取 Top-N 条送入 Cross-Encoder Reranker，逐对计算 `(query, chunk)` 相关性分数，重新排序后取前 12。支持两种后端：

| 后端 | 模型 | 说明 |
|------|------|------|
| DashScope API | `gte-rerank-v2` | 阿里云在线 API，免部署，自带免费额度 |
| 本地 | `BAAI/bge-reranker-base` | HuggingFace Cross-Encoder，离线可用 |

### 动态 TopK

并非固定取 K 个父块，而是根据 query 长度和 reranker 分数分布自动调节：

- 长 query (≥36 字符)：上下文需求更高，自动 +2
- 分数高度集中：说明主题聚焦，适度增加以覆盖细节
- 最终范围：4-8 块，兼顾回答完整性与 token 成本

### 答案溯源

每个回答末尾附带 **参考来源** 列表，标注文件名和原文摘录。可在 Gradio 界面中点击文档直接查看全文。

### 双模式切换

通过 `.env` 中的两行配置即可在 API 模式与本地模式间切换：

| 组件 | API 模式 | 本地模式 |
|------|----------|----------|
| 嵌入 | DashScope `text-embedding-async-v2` | BGE-M3 (~2.2GB) |
| 重排序 | DashScope `gte-rerank-v2` | BGE-Reranker (~1.0GB) |
| 生成 | Qwen-Max | — |

API 模式零模型下载即可运行，适合快速体验；本地模式完全离线，适合生产环境或隐私敏感场景。

---

## 快速开始

### 前置要求

- Python 3.10+
- [DashScope API Key](https://dashscope.aliyuncs.com/) (阿里云百炼，免费额度充足)

### 1. 克隆与安装

```bash
git clone git@github.com:lhh737/-RAG-Chinese-QA.git
cd -- -RAG-Chinese-QA           # 目录名以 - 开头，需用 -- 转义

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入：

```ini
DASHSCOPE_API_KEY=sk-your-real-key-here
```

其他配置保持默认即可（API 模式，qwen-max 推理）。

### 3. 启动

```bash
python mvp_app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传文档即可开始问答。

---

## 使用指南

### 两个界面版本

| 入口 | 端口 | 说明 |
|------|------|------|
| `mvp_app.py` | 7860 | 精简版：上传 → 对话，轻量快速 |
| `gradio_app.py` | 7862 | 完整版：包含论文目录批量导入、文档内容查看 |

### 脚本工具

```bash
# 下载 arXiv 论文（按主题批量获取 2024-2026 AI 论文）
python scripts/fetch_papers.py --max-per-topic 10 --topic rag

# 批量导入 papers/ 到知识库
python scripts/batch_import.py

# 下载本地模型（切换离线模式前执行）
python scripts/download_bge.py
```

### 切换到本地模型

```bash
# 1. 下载模型（一次性，约 3.2GB）
python scripts/download_bge.py

# 2. 修改 .env
EMBED_MODE=local
EMBED_LOCAL_MODEL=models/bge-m3
RERANK_MODE=local
RERANK_LOCAL_MODEL=models/bge-reranker-base

# 3. 重启应用
python mvp_app.py
```

### 配置调参

`config/faiss.yml` 中可调整检索参数：

```yaml
vector_fetch_k: 24      # 向量路召回量
bm25_fetch_k: 24        # BM25 路召回量
rerank_top_n: 12        # 重排序后保留
base_context_k: 4       # 基础上下文块数
max_context_k: 8        # 动态上限
parent_chunk_size: 1200 # 父块大小
child_chunk_size: 400   # 子块大小
```

---

## 项目结构

```
├── config/
│   ├── settings.py          # 环境变量解析，所有配置项入口
│   ├── faiss.yml            # 检索参数调优
│   └── rag.yml              # 模型与生成参数
│
├── model/
│   ├── factory.py           # 模型工厂：API/本地 双模式自动切换
│   └── dashscope_embedding.py  # DashScope Embedding API 封装
│
├── rag/
│   ├── pipeline.py          # 统一管线入口 RAGPipeline
│   ├── hybrid_retriever.py  # 混合检索引擎 + 动态 TopK
│   ├── hyde.py              # HyDE 假设性文档生成
│   ├── vector_store.py      # FAISS 索引 + 父子块管理 + 文件清单
│   ├── document_loader.py   # PDF/TXT 解析 + 父子分块
│   └── generator.py         # Qwen API 生成封装
│
├── scripts/
│   ├── fetch_papers.py      # arXiv 论文批量下载
│   ├── batch_import.py      # 批量导入文档到知识库
│   ├── download_bge.py      # BGE 本地模型下载器
│   └── download_models.py   # 通用模型下载
│
├── prompts/
│   └── rag_summarize.txt    # RAG 生成 Prompt 模板
│
├── utils/                   # 工具：日志、配置解析、路径
├── data/                    # 文档数据 + 论文存储
│   └── papers/              # 知识库论文 PDF
│
├── mvp_app.py               # Gradio 精简版入口
├── gradio_app.py            # Gradio 完整版入口
└── requirements.txt
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | LangChain 1.2 |
| 向量库 | FAISS |
| 嵌入 | DashScope `text-embedding-async-v2` / BGE-M3 |
| 稀疏检索 | BM25Okapi + jieba 分词 |
| 重排序 | DashScope `gte-rerank-v2` / `BAAI/bge-reranker-base` |
| 生成 | Qwen-Max (OpenAI 兼容 API) |
| 文档解析 | PyPDFLoader + RecursiveCharacterTextSplitter |
| 界面 | Gradio 5 |

## License

MIT
