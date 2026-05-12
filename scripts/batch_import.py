"""
批量导入工具：扫描 data/papers/ 下所有 PDF/TXT，批量索引到向量库。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline


def collect_papers(root: str) -> list[str]:
    papers = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith((".pdf", ".txt")) and not f.startswith("."):
                papers.append(os.path.join(dirpath, f))
    return sorted(papers)


def batch_import(root: str = "data/papers"):
    import sys
    p = RAGPipeline(use_hybrid=True, use_hyde=True)
    papers = collect_papers(root)

    if not papers:
        print(f"[batch] 在 {root} 下未找到 PDF/TXT 文件")
        return

    print(f"[batch] 发现 {len(papers)} 篇论文，开始导入...")
    sys.stdout.flush()
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, fp in enumerate(papers, 1):
        ok, msg = p.upload_and_index(fp)
        if ok:
            ok_count += 1
            print(f"  [{i}/{len(papers)}] ✓ {os.path.basename(fp)} — {msg}")
        elif "已存在" in msg:
            skip_count += 1
            print(f"  [{i}/{len(papers)}] - {os.path.basename(fp)} — 已跳过（重复）")
        else:
            fail_count += 1
            print(f"  [{i}/{len(papers)}] ✗ {os.path.basename(fp)} — {msg}")
        sys.stdout.flush()

    print(f"\n[batch] 完成：成功 {ok_count}，跳过 {skip_count}，失败 {fail_count}")


if __name__ == "__main__":
    batch_import()
