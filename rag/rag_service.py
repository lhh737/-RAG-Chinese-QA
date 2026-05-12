"""
RAG 总结服务：混合检索 + Qwen API 生成 + 答案溯源（参考资料编号）。
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate

from rag.generator import generate
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStore
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStore()
        self.hybrid = HybridRetriever(self.vector_store)
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

    def retriever_docs(self, query: str):
        """兼容旧接口：返回向量检索子块（不含混合逻辑）。"""
        if self.vector_store.faiss_store is None:
            return []
        return self.vector_store.faiss_store.similarity_search(query, k=8)

    def rag_summarize_with_sources(self, query: str) -> dict[str, Any]:
        context, provenance = self.hybrid.retrieve_for_rag(query)
        if not context.strip():
            return {
                "answer": "当前知识库中未找到与问题相关的资料，请先上传文档后再试。",
                "sources": [],
            }
        prompt_str = self.prompt_template.format(input=query, context=context)
        # 使用 Qwen API 生成，传递空字符串作为 system prompt（已编码在 prompt template 中）
        answer = generate("", context=context, question=query)
        return {"answer": answer, "sources": provenance}

    def rag_summarize(self, query: str) -> str:
        """供调用：返回带溯源列表的纯文本。"""
        out = self.rag_summarize_with_sources(query)
        lines = [out["answer"]]
        if out.get("sources"):
            lines.append("\n--- 参考来源 ---")
            for i, s in enumerate(out["sources"], start=1):
                src = s.get("source") or ""
                excerpt = s.get("excerpt", "")[:80]
                lines.append(f"{i}. {src} — {excerpt}…")
        return "\n".join(lines)


if __name__ == "__main__":
    import os

    os.environ.setdefault("RAG_SMOKE_TEST", "1")
    rag = RagSummarizeService()
    print(rag.rag_summarize("什么是Transformer注意力机制"))
