"""
直接下载重要 AI 经典论文（按 arxiv ID 直连 PDF，跳过搜索 API）。
"""
from __future__ import annotations

import os, sys, time, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPERS_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "papers")

HEADERS = {"User-Agent": "RAG-Project/1.0"}

# 按主题整理的重要论文（id, 年份, 标题）
PAPERS = {
    "llm_foundation": [
        ("1706.03762", "2017", "Attention Is All You Need"),
        ("2005.14165", "2020", "Language Models are Few-Shot Learners (GPT-3)"),
        ("2205.01068", "2022", "Training language models to follow instructions (InstructGPT)"),
        ("2302.13971", "2023", "LLaMA: Open and Efficient Foundation Language Models"),
        ("2303.08774", "2023", "GPT-4 Technical Report"),
        ("2310.06825", "2023", "Mistral 7B"),
        ("2312.11805", "2023", "Mixtral of Experts"),
        ("2401.14295", "2024", "The Llama 3 Herd of Models"),
        ("2405.04434", "2024", "Phi-3 Technical Report"),
        ("2204.05862", "2022", "Training Compute-Optimal Large Language Models (Chinchilla)"),
    ],
    "rag": [
        ("2005.11401", "2020", "RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
        ("2312.10997", "2023", "RAG Survey: Retrieval-Augmented Generation for Large Language Models"),
        ("2402.19473", "2024", "Self-RAG: Learning to Retrieve, Generate, and Critique"),
        ("2401.15884", "2024", "RAFT: Adapting Language Model to Domain Specific RAG"),
        ("2405.07437", "2024", "Corrective Retrieval Augmented Generation"),
        ("2403.14403", "2024", "MM-RAG: Multi-Modal Retrieval Augmented Generation"),
        ("2405.05342", "2024", "Evaluating RAG with RAGAS"),
        ("2402.07403", "2024", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"),
        ("2401.05856", "2024", "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study"),
        ("2309.01431", "2023", "CRAG: Comprehensive RAG Benchmark"),
    ],
    "agent": [
        ("2308.11432", "2023", "Generative Agents: Interactive Simulacra of Human Behavior"),
        ("2309.07864", "2023", "Toolformer: Language Models Can Teach Themselves to Use Tools"),
        ("2305.17544", "2023", "Gorilla: Large Language Model Connected with Massive APIs"),
        ("2402.05119", "2024", "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"),
        ("2309.11495", "2024", "ToRA: A Tool-Integrated Reasoning Agent"),
        ("2310.04406", "2023", "AgentBench: Evaluating LLMs as Agents"),
        ("2401.00812", "2024", "AutoGPT: A Benchmark for Automated Task Completion"),
        ("2310.11569", "2023", "WebArena: A Realistic Web Environment for Building Autonomous Agents"),
        ("2402.12317", "2024", "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering"),
        ("2407.01561", "2024", "OpenHands: An Open Platform for AI Software Developers as Generalist Agents"),
    ],
}


def download_pdf(arxiv_id: str, target_dir: str) -> bool:
    """下载单篇论文 PDF。"""
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(target_dir, filename)
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 10000:
        return True  # 已存在

    # 尝试多个镜像
    urls = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
    ]

    for url in urls:
        for attempt in range(2):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=60)
                resp.raise_for_status()
                if len(resp.content) > 10000:
                    os.makedirs(target_dir, exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    print(f"  ✓ {arxiv_id}")
                    return True
            except Exception as e:
                if attempt == 0:
                    time.sleep(5)
    print(f"  ✗ {arxiv_id} 下载失败")
    return False


def main():
    total = 0
    for topic, papers in PAPERS.items():
        target_dir = os.path.join(PAPERS_BASE, topic)
        os.makedirs(target_dir, exist_ok=True)
        existing = [f.split(".")[0] for f in os.listdir(target_dir) if f.endswith(".pdf")]
        print(f"\n{'='*50}")
        print(f"{topic}: 已有 {len(existing)} 篇")
        for arxiv_id, year, title in papers:
            if arxiv_id in existing:
                print(f"  = {arxiv_id} 已存在，跳过")
                continue
            if download_pdf(arxiv_id, target_dir):
                total += 1
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"新增下载: {total} 篇")

    # 统计总数
    import glob
    all_pdfs = glob.glob(os.path.join(PAPERS_BASE, "**", "*.pdf"), recursive=True)
    print(f"论文总数: {len(all_pdfs)} 篇")


if __name__ == "__main__":
    main()
