"""
最小 MVP 测试：索引 2 篇文档，跑一次问答。
"""
import sys
sys.path.insert(0, ".")

from rag.pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("RAG 最小 MVP 测试")
    print("=" * 60)

    # 关闭 HyDE，减少 API 调用，纯测 RAG 链路
    p = RAGPipeline(use_hybrid=True, use_hyde=False)

    # 索引两篇测试文档
    docs = [
        "data/llm_intro.txt",
        "data/test_transformer.txt",
    ]

    for fp in docs:
        ok, msg = p.upload_and_index(fp)
        print(f"[索引] {fp} -> {msg}")

    # 看看知识库里有什么
    files = p.get_file_list()
    print(f"\n知识库共 {len(files)} 篇文档:")
    for f in files:
        print(f"  - {f['filename']} ({f['size_kb']}KB)")

    # 问答测试
    questions = [
        "什么是RAG？它有什么优点？",
        "Transformer的核心机制是什么？",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"[问] {q}")
        answer = p.query(q)
        print(f"[答] {answer}")

    print(f"\n{'=' * 60}")
    print("MVP 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
