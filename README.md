# RAG-Chinese-QA

基于 LangChain 的 RAG 中文文档问答系统，支持混合检索、HyDE 增强与 BGE 重排序。

## 功能特性

- **混合检索** — FAISS 向量检索 + BM25 稀疏检索，互补召回
- **HyDE 增强** — 先让 LLM 生成假设性答案，再以此为 query 检索，提升召回率
- **重排序** — 支持 DashScope API (`gte-rerank-v2`) 或本地 `BAAI/bge-reranker-base`
- **父子分块** — 子块精准检索，父块完整上下文还原
- **动态 TopK** — 根据 query 长度和 reranker 分数分布自适应调节上下文块数
- **答案溯源** — 每个回答附带参考文档来源
- **Gradio Web 界面** — 支持文件上传、批量导入、对话、文档查看

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | LangChain |
| 向量库 | FAISS |
| 嵌入 | DashScope `text-embedding-async-v2` / BGE-M3（本地） |
| 重排序 | DashScope `gte-rerank-v2` / BGE-Reranker（本地） |
| 生成 | Qwen（DashScope 兼容 API） |
| 检索 | BM25（jieba 分词）+ 向量 + Reranker |
| 界面 | Gradio |

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone git@github.com:lhh737/-RAG-Chinese-QA.git
cd RAG-Chinese-QA

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填写你的 DASHSCOPE_API_KEY
```

### 3. 运行

```bash
# 启动 Gradio Web 界面
python mvp_app.py
```

默认使用 API 模式（DashScope），无需下载模型即可运行。切换到本地模式请修改 `.env` 中的 `EMBED_MODE` 和 `RERANK_MODE`。

## 项目结构

```
├── config/             # 配置文件（YAML / settings.py）
├── data/               # 文档数据目录
├── model/              # 模型工厂（API + 本地切换）
├── prompts/            # Prompt 模板
├── rag/                # RAG 核心管线
│   ├── pipeline.py     # 统一 RAG 管线入口
│   ├── hybrid_retriever.py  # 混合检索 + 重排序
│   ├── hyde.py         # HyDE 假设性文档生成
│   ├── vector_store.py # FAISS 向量库管理
│   ├── generator.py    # LLM 生成
│   └── document_loader.py  # 文档解析与分块
├── utils/              # 工具函数
├── mvp_app.py          # Gradio 应用入口
├── gradio_app.py       # Gradio 扩展版（含论文批量导入）
└── requirements.txt
```

## License

MIT
