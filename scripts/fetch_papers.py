"""
arXiv 论文批量下载器：按主题搜索 2024-2026 年 AI 高质量论文，自动分类到 data/papers/。

用法：
  python scripts/fetch_papers.py                     # 下载全部主题（每主题 10 篇）
  python scripts/fetch_papers.py --max-per-topic 20  # 每主题下载 20 篇
  python scripts/fetch_papers.py --topic rag         # 仅下载 RAG 主题
  python scripts/fetch_papers.py --dry-run           # 只查看搜索到的论文，不下载
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARXIV_API = "http://export.arxiv.org/api/query"

PAPERS_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "papers",
)

# ── 搜索主题配置 ──────────────────────────────────────────
# 每个主题对应 data/papers/ 下的子目录 + arxiv 搜索 query
TOPICS = {
    "llm_foundation": {
        "query": "(cat:cs.CL OR cat:cs.LG) AND (large language model* OR foundation model* OR transformer* OR GPT)",
        "description": "LLM 基础模型论文",
    },
    "rag": {
        "query": '(cat:cs.CL OR cat:cs.IR OR cat:cs.AI) AND ("retrieval augmented generation" OR RAG OR "retrieval-augmented")',
        "description": "RAG 检索增强生成",
    },
    "agent": {
        "query": '(cat:cs.AI OR cat:cs.MA OR cat:cs.CL) AND ("AI agent*" OR "LLM agent*" OR "autonomous agent*" OR "tool use" OR function-calling)',
        "description": "AI Agent 智能体",
    },
    "embedding": {
        "query": '(cat:cs.CL OR cs:IR) AND ("text embedding*" OR "sentence embedding*" OR "representation learning" OR BGE OR "text2vec")',
        "description": "文本嵌入/表征学习",
    },
    "alignment": {
        "query": '(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND ("RLHF" OR "DPO" OR "alignment" OR "safety" OR "constitutional AI" OR "preference optimization")',
        "description": "AI 对齐与安全",
    },
    "inference": {
        "query": '(cat:cs.CL OR cat:cs.LG OR cs:AR) AND ("model compression" OR quantization OR distillation OR pruning OR "efficient inference" OR "speculative decoding")',
        "description": "模型推理优化",
    },
    "multimodal": {
        "query": '(cat:cs.CV OR cat:cs.CL OR cat:cs.MM OR cat:cs.AI) AND ("multimodal" OR "vision-language" OR CLIP OR LLaVA OR "visual instruction")',
        "description": "多模态模型",
    },
    "prompting": {
        "query": '(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND ("prompt engineering" OR "chain-of-thought" OR "in-context learning" OR "prompt design" OR "few-shot")',
        "description": "提示工程与 In-Context Learning",
    },
    "evaluation": {
        "query": '(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND ("benchmark*" OR "evaluation" OR "LLM eval" OR "model benchmark*" OR "dataset")',
        "description": "LLM 评估与基准",
    },
    "knowledge_base": {
        "query": '(cat:cs.AI OR cat:cs.CL OR cat:cs.IR) AND ("knowledge graph*" OR "knowledge base*" OR "factual" OR "knowledge enhancement" OR "retrieval")',
        "description": "知识库与知识图谱",
    },
}

HEADERS = {
    "User-Agent": "RAG-Project/1.0 (mailto:example@example.com)",
}

QUERY_INTERVAL = 3  # arXiv API 要求至少 3 秒间隔


def arxiv_search(query: str, max_results: int = 20, start: int = 0) -> list[dict]:
    """调用 arXiv API 搜索论文，返回论文信息列表。"""
    params = {
        "search_query": query,
        "start": start,
        "max_results": min(max_results, 100),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  [arxiv] 请求失败（第{attempt+1}次）: {e}")
            time.sleep(5)
    else:
        return []

    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    papers = []
    for entry in root.findall("a:entry", ns):
        paper_id = entry.find("a:id", ns).text.strip().split("/")[-1].split("v")[0]
        title = entry.find("a:title", ns).text.strip().replace("\n", " ").replace("\r", "")
        summary = entry.find("a:summary", ns).text.strip().replace("\n", " ").replace("\r", "")[:500]

        authors = []
        for author in entry.findall("a:author", ns):
            name = author.find("a:name", ns)
            if name is not None:
                authors.append(name.text)

        published = entry.find("a:published", ns)
        published_date = published.text[:10] if published is not None else ""

        # 提取分类
        categories = [cat.get("term", "") for cat in entry.findall("a:category", ns)]

        # 提取 PDF 链接
        pdf_link = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href", "")
                break

        # arXiv 的 HTML 摘要页
        abs_url = ""
        for link in entry.findall("a:link", ns):
            if link.get("rel") == "alternate":
                abs_url = link.get("href", "")
                break

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published_date,
                "categories": categories,
                "pdf_url": pdf_link,
                "abs_url": abs_url,
            }
        )

    return papers


def _is_english(text: str) -> bool:
    """粗略判断是否为英文论文。"""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    return chinese_chars / max(len(text), 1) < 0.1


def classify_paper(title: str, summary: str, categories: list[str]) -> str | None:
    """根据标题和摘要分类到 topic key。"""
    text = (title + " " + summary).lower()

    # 使用分类标签优先
    cat_set = set(categories)
    if any("cs.CV" in c or "cs.MM" in c for c in cat_set):
        patterns = ["multimodal", "vision-language", "visual", "image", "video"]
        if any(p in text for p in patterns):
            return "multimodal"

    # 关键词匹配
    rules = [
        ("agent", ["agent", "function-calling", "tool use", "tool-use"]),
        ("alignment", ["rlhf", "dpo", "alignment", "constitutional", "preference optimization", "safety"]),
        ("embedding", ["embedding", "representation learning", "sentence", "text2vec"]),
        ("evaluation", ["benchmark", "evaluation", "llm eval", "model benchmark"]),
        ("inference", ["quantization", "distillation", "pruning", "efficient inference", "speculative decoding", "compression"]),
        ("knowledge_base", ["knowledge graph", "knowledge base", "knowledge enhancement", "factual"]),
        ("multimodal", ["multimodal", "vision-language", "cllp", "llava", "visual instruction", "image", "video"]),
        ("prompting", ["chain-of-thought", "prompt engineering", "in-context learning", "few-shot", "prompt design"]),
        ("rag", ["retrieval augmented generation", "rag", "retrieval-augmented", "retrieval", "dense retrieval"]),
        ("llm_foundation", ["large language model", "foundation model", "transformer", "gpt", "llm"]),
    ]

    for topic, keywords in rules:
        if any(kw in text for kw in keywords):
            return topic

    return None


def get_existing_ids() -> set[str]:
    """获取已下载论文的 arxiv ID 集合，避免重复下载。"""
    existing: set[str] = set()
    papers_dir = Path(PAPERS_BASE)
    if not papers_dir.exists():
        return existing
    for f in papers_dir.rglob("*.pdf"):
        for part in f.stem.split("__"):
            part = part.strip()
            # arxiv id 通常是纯数字 + 小数点 + 数字
            if part and (part.startswith("2") or "." in part):
                existing.add(part)
    return existing


def download_pdf(paper: dict, target_dir: str) -> bool:
    """下载单篇论文 PDF。"""
    pdf_url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{paper['id']}.pdf"
    arxiv_id = paper["id"]

    # 用 arxiv ID 命名文件，保持兼容
    filename = f"{arxiv_id}__{paper['published']}.pdf"
    filepath = os.path.join(target_dir, filename)

    if os.path.isfile(filepath):
        return True

    for attempt in range(3):
        try:
            resp = requests.get(pdf_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            os.makedirs(target_dir, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(resp.content)
            if os.path.getsize(filepath) > 10000:  # 至少 10KB
                print(f"    ✓ {arxiv_id} ({paper['published']})")
                time.sleep(0.5)
                return True
            else:
                os.remove(filepath)
                print(f"    ✗ {arxiv_id} 文件太小，重试...")
        except Exception as e:
            print(f"    ✗ {arxiv_id} 下载失败: {e}")
            time.sleep(3)
    return False


def process_topic(
    topic_key: str,
    topic_config: dict,
    max_per_topic: int,
    dry_run: bool,
) -> tuple[int, int]:
    """处理单个主题：搜索 + 下载。"""
    query_str = topic_config["query"]
    existing_ids = get_existing_ids()
    all_new = 0
    all_fail = 0
    target_dir = os.path.join(PAPERS_BASE, topic_key)
    start = 0
    collected_papers = []

    print(f"\n{'='*60}")
    print(f"主题: {topic_config['description']} [{topic_key}]")
    print(f"目录: {target_dir}")

    while len(collected_papers) < max_per_topic * 3 and start < 200:
        results = arxiv_search(query_str, max_results=50, start=start)
        if not results:
            break

        for p in results:
            if p["id"] in existing_ids:
                continue
            if not _is_english(p["title"]):
                continue

            # 再次确认分类正确
            assigned = classify_paper(p["title"], p["summary"], p["categories"])
            if assigned is None or assigned == topic_key:
                collected_papers.append(p)

        start += 50
        time.sleep(QUERY_INTERVAL)
        print(f"  [arxiv] 已搜索 {start} 条，收集 {len(collected_papers)} 篇候选")

    # 按日期倒排取最新的
    collected_papers.sort(key=lambda x: x["published"], reverse=True)
    selected = collected_papers[:max_per_topic]

    if dry_run:
        print(f"\n  [干运行] 共 {len(selected)} 篇论文待下载：")
        for p in selected:
            print(f"    {p['published']} [{p['id']}] {p['title'][:70]}")
        return 0, 0

    os.makedirs(target_dir, exist_ok=True)

    def _dl_one(p):
        return download_pdf(p, target_dir)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_dl_one, p): p for p in selected}
        for future in as_completed(futures):
            if future.result():
                all_new += 1
            else:
                all_fail += 1

    return all_new, all_fail


def main():
    parser = argparse.ArgumentParser(description="arXiv 论文批量下载器")
    parser.add_argument("--max-per-topic", type=int, default=10, help="每主题下载篇数（默认 10）")
    parser.add_argument("--topic", choices=list(TOPICS.keys()) + ["all"], default="all", help="指定主题")
    parser.add_argument("--dry-run", action="store_true", help="干运行，只显示搜索结果不下载")
    args = parser.parse_args()

    topics = TOPICS.items() if args.topic == "all" else [(args.topic, TOPICS[args.topic])]

    total_new = 0
    total_fail = 0
    for key, config in topics:
        ok, fail = process_topic(key, config, args.max_per_topic, args.dry_run)
        total_new += ok
        total_fail += fail

    print(f"\n{'='*60}")
    if args.dry_run:
        print("干运行结束，以上为搜索结果预览。移除 --dry-run 实际下载。")
    else:
        print(f"下载完成：成功 {total_new}，失败 {total_fail}")
        print(f"论文保存在 {PAPERS_BASE}")
        if total_fail > 0:
            print("失败的论文可重试：python scripts/fetch_papers.py --topic <topic>")
        print("\n下一步：python scripts/batch_import.py  # 导入知识库")


if __name__ == "__main__":
    main()
