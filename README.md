<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  基于 LangChain 的中文 RAG 文档问答系统<br>
  混合检索 · HyDE 增强 · 重排序 · 答案溯源
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/framework-LangChain%201.2-orange" alt="LangChain">
</p>

---

## 概述

上传 PDF 或 TXT 文档，用自然语言提问，获得带来源标注的精确回答。

系统将文档切分为 400 字子块用于检索、1200 字父块用于上下文还原；检索时 FAISS 向量与 BM25 关键词两路并行召回，经 Cross-Encoder 重排序后动态选取 4-8 个最相关段落，交由 Qwen-Max 生成答案并附录引用来源。

**API 模式零模型下载，拿到 API Key 即可运行**；也支持完全离线的本地 BGE 模型。

---

## 快速开始

### 前置

- Python 3.10+
- 阿里云百炼 [DashScope API Key](https://dashscope.aliyuncs.com/)（注册即送免费额度）

### 安装

```bash
git clone git@github.com:lhh737/-RAG-Chinese-QA.git
cd -- -RAG-Chinese-QA

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，修改一行：

```ini
DASHSCOPE_API_KEY=sk-你的真实Key
```

### 运行

```bash
python mvp_app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传文档 → 输入问题 → 获取答案。

---

## 检索管线

```
用户提问
  │
  ├─→ HyDE         LLM 生成假设性答案，替换原始 query 以提升召回
  │
  ├─→ 粗筛          FAISS 向量检索 (×24)  ─┐
  │                BM25 稀疏检索 (×24)  ─┤  合并去重，~50 条候选
  │                                       ┘
  ├─→ 精排          Cross-Encoder / gte-rerank-v2 → Top 12
  │
  ├─→ 还原          子块 → 父块映射，动态 TopK (4-8 块)
  │
  └─→ 生成          Qwen-Max + 上下文 → 答案 + 溯源引用
```

---

## 功能详解

### 混合检索：FAISS + BM25

| 路径 | 原理 | 擅长 |
|------|------|------|
| FAISS 向量 | 稠密语义向量余弦相似度 | 同义表达、概念匹配 |
| BM25 稀疏 | jieba 分词 + 倒排索引 | 术语精确命中、专有名词 |

两路各召回 24 条后合并去重，有效降低单一检索的漏召风险。

### HyDE：假设性文档增强

用户问题通常短且口语化，直接检索效果有限。HyDE 先调用 LLM 将问题扩展为一段"假想文档"——例如问 "transformer 为什么用多头注意力" 时，LLM 会生成一段教科书风格的技术描述，这段描述与真实文档的语义空间更接近，从而显著提升检索命中率。LLM 调用失败时自动回退为原始 query。

### 父子分块

检索精度和上下文完整性往往不可兼得——小块检索准但缺乏上下文，大块语义完整但噪声多。父子分块策略将两者解耦：

- **子块 (~400 字)**：送入 FAISS 索引和 BM25 语料库，负责精准匹配
- **父块 (~1200 字)**：检索命中子块后，向上关联到所属父块，将完整段落提供给 LLM

### 重排序

候选池合并后送入 Cross-Encoder Reranker 逐对打分重排：

| 后端 | 模型 | 特点 |
|------|------|------|
| DashScope API | `gte-rerank-v2` | 免部署，免费额度 100 万 token |
| 本地 Cross-Encoder | `BAAI/bge-reranker-base` | 完全离线，~1GB 显存 |

### 动态 TopK

送给 LLM 的上下文块数量不是固定的，而是根据 query 长度和 reranker 分数分布自动调节（4-8 块），在回答完整性和 token 消耗之间平衡。

### 双模式切换

`.env` 中两行配置即可切换：

```ini
# API 模式（默认，零模型下载）
EMBED_MODE=api
RERANK_MODE=api

# 本地模式（离线可用，需先执行 scripts/download_bge.py）
EMBED_MODE=local
RERANK_MODE=local
```

---

## 使用指南

### 界面版本

| 入口 | 端口 | 场景 |
|------|------|------|
| `mvp_app.py` | 7860 | 日常使用：上传对话一体，轻量快速 |
| `gradio_app.py` | 7862 | 研究场景：含论文批量导入、文档内容查看、DataFrame 文件列表 |

### 脚本工具

```bash
# 从 arXiv 批量下载论文（按主题，自动去重，支持断点续传）
python scripts/fetch_papers.py --topic rag --max-per-topic 20

# 将 papers/ 目录全部导入知识库
python scripts/batch_import.py

# 下载 BGE 本地模型（约 3.2GB，切换离线模式前运行一次）
python scripts/download_bge.py
```

### 检索参数调优

`config/faiss.yml`：

```yaml
vector_fetch_k: 24       # 向量召回数
bm25_fetch_k: 24         # BM25 召回数
rerank_top_n: 12         # 重排序后截断
base_context_k: 4        # 基础上下文块数
max_context_k: 8         # 动态上限
parent_chunk_size: 1200  # 父块字符数
child_chunk_size: 400    # 子块字符数
```

---

## 项目结构

```
├── config/
│   ├── settings.py           # 环境变量解析，所有配置入口
│   ├── faiss.yml             # 检索超参
│   └── rag.yml               # 模型与生成参数
├── model/
│   ├── factory.py            # 模型工厂：API/本地自动切换 + LRU 缓存
│   └── dashscope_embedding.py
├── rag/
│   ├── pipeline.py           # 统一管线 RAGPipeline
│   ├── hybrid_retriever.py   # 混合检索 + 动态 TopK
│   ├── hyde.py               # HyDE 假设文档生成
│   ├── vector_store.py       # FAISS 索引 + 父子块 + manifest
│   ├── document_loader.py    # PDF/TXT 解析 + 父子分块
│   └── generator.py          # Qwen API 封装
├── scripts/
│   ├── fetch_papers.py       # arXiv 论文批量下载器
│   ├── batch_import.py       # 文档批量导入
│   ├── download_bge.py       # BGE 模型下载
│   └── download_models.py    # 通用模型下载
├── prompts/rag_summarize.txt # RAG 生成 Prompt 模板
├── utils/                    # 日志、配置解析、路径工具
├── data/papers/              # 知识库论文
├── mvp_app.py                # Gradio 入口（精简版）
├── gradio_app.py             # Gradio 入口（完整版）
└── requirements.txt
```

---

## 常见问题

<details>
<summary><b>如何获取 DashScope API Key？</b></summary>

访问 [阿里云百炼控制台](https://dashscope.aliyuncs.com/)，注册后进入 API-KEY 管理页面创建 Key。新用户赠送免费额度（text-embedding-async-v2 2000 万 token / gte-rerank-v2 100 万 token / qwen-max 100 万 token）。
</details>

<details>
<summary><b>支持哪些文档格式？</b></summary>

PDF 和 TXT。PDF 通过 PyPDFLoader 提取文本，扫描版 PDF（纯图片）需要先 OCR 处理。
</details>

<details>
<summary><b>切换嵌入模型后旧索引报错怎么办？</b></summary>

系统会自动检测 FAISS 维度与当前嵌入模型是否匹配，不匹配时重建索引。如果仍有问题，删除 `faiss_db/` 目录后重启即可。
</details>

<details>
<summary><b>上传大文档很慢？</b></summary>

瓶颈通常在嵌入 API（DashScope 免费额度下 QPS 有限）。可以调大 `child_chunk_size` 减少分块数以加速，但会损失一定检索精度。
</details>

## 许可证

MIT — 可自由使用、修改和分发。
