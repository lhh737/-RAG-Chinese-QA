"""
RAG MVP v2 — 懒加载 + 实时文档列表 + API 优先
"""
from __future__ import annotations

import os, json, shutil, glob
import gradio as gr

from config.settings import UPLOAD_DIR, PAPERS_DIR, FAISS_PERSIST_DIR

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline(use_hybrid=True, use_hyde=False)
    return _pipeline


# ── 文档列表（只读 manifest，不触发任何模型加载）───────────

def _read_manifest() -> list[dict]:
    """纯读 JSON，不加载 pipeline。"""
    path = os.path.join(FAISS_PERSIST_DIR, "manifest.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return sorted(
            json.load(f).values(),
            key=lambda x: x.get("loaded_at", ""),
            reverse=True,
        )


def render_doc_list() -> tuple[str, gr.Dropdown]:
    """返回 markdown 文档列表 + 下拉选项。"""
    docs = _read_manifest()
    if not docs:
        return "### 已载入文档\n\n*（暂无文档）*", gr.Dropdown(choices=[], value=None, interactive=True)

    lines = ["### 已载入文档\n"]
    choices = []
    for i, d in enumerate(docs, 1):
        lines.append(f"{i}. **{d['filename']}**  ({d['size_kb']}KB)  \n"
                     f"   _{d['loaded_at']}_")
        choices.append(f"{d['filename']} ({d['loaded_at']})")

    return "\n".join(lines), gr.Dropdown(choices=choices, value=choices[0], interactive=True)


def scan_for_import() -> list[str]:
    """扫描可导入的文件路径（不重复的）。"""
    imported = set()
    for d in _read_manifest():
        imported.add(d.get("filename", ""))
        imported.add(d.get("path", ""))

    found = set()
    for pat in [
        os.path.join(PAPERS_DIR, "**", "*.pdf"),
        os.path.join(PAPERS_DIR, "**", "*.txt"),
        "data/*.txt",
        "data/*.pdf",
    ]:
        for fp in glob.glob(pat, recursive=True):
            if not os.path.isfile(fp):
                continue
            name = os.path.basename(fp)
            if name.startswith("."):
                continue
            if fp in imported or name in imported:
                continue
            found.add(fp)
    return sorted(found)


# ── 业务操作 ────────────────────────────────────────────

def do_import(file_path: str) -> tuple[str, str, gr.Dropdown]:
    """导入一个文件到知识库。"""
    if not file_path or not os.path.isfile(file_path):
        return f"文件不存在: {file_path}", *render_doc_list()

    p = get_pipeline()
    ok, msg = p.upload_and_index(file_path)
    doc_md, dd = render_doc_list()
    return msg, doc_md, dd


def do_upload(file_obj) -> tuple[str, str, gr.Dropdown]:
    """上传 + 导入。"""
    if file_obj is None:
        return "请选择文件", *render_doc_list()

    src = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    filename = os.path.basename(src)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, filename)
    shutil.copy2(src, dest)
    return do_import(dest)


def do_chat(message: str, history: list):
    """对话入口。"""
    history = history or []
    if not message or not message.strip():
        return history, ""

    p = get_pipeline()
    if p.vector_store.is_empty:
        history.append((message, "知识库为空，请先导入文档。"))
        return history, ""

    try:
        result = p.query_with_sources(message.strip())
        answer = result["answer"]
        sources = result.get("sources", [])
        if sources:
            lines = ["\n\n**参考来源**"]
            seen = set()
            for s in sources:
                src = s.get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    lines.append(f"- {src}")
            answer += "\n".join(lines)
    except Exception as e:
        answer = f"生成回答时出错: {e}"

    history.append((message, answer))
    return history, ""


def refresh_all() -> tuple[str, gr.Dropdown]:
    """刷新文档列表 + 可导入文件列表。"""
    doc_md, dd = render_doc_list()
    files = scan_for_import()
    return doc_md, dd, gr.Dropdown(choices=files, value=files[0] if files else None, interactive=True)


# ── UI ────────────────────────────────────────────────────

CSS = """
.doc-scroll { max-height: 320px; overflow-y: auto; padding: 8px; border: 1px solid #444; border-radius: 6px; }
"""


def build_ui():
    with gr.Blocks(title="RAG 文档问答", css=CSS) as demo:
        gr.Markdown("## RAG 文档问答系统  \n"
                    "嵌入/重排序: DashScope API | LLM: qwen-max | 懒加载")

        with gr.Row():
            # ═══ 左侧 ═══
            with gr.Column(scale=1):
                # 上传区
                gr.Markdown("### 上传文档")
                upload_file = gr.File(label="选文件（txt/pdf）", file_types=[".txt", ".pdf"])
                upload_btn = gr.Button("上传并导入", variant="primary", size="sm")
                status_text = gr.Textbox(label="状态", interactive=False, show_label=False)

                # 已有文件导入区
                gr.Markdown("### 从 data/ 导入")
                available = scan_for_import()
                file_picker = gr.Dropdown(
                    label="可导入文件",
                    choices=available,
                    value=available[0] if available else None,
                    interactive=True,
                )
                import_btn = gr.Button("导入选中文件", variant="secondary", size="sm")

                # 已载入文档列表
                doc_md, doc_sel = render_doc_list()
                doc_display = gr.Markdown(doc_md, elem_classes=["doc-scroll"])

            # ═══ 右侧 ═══
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=420, label="对话（含答案溯源）")
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入问题",
                        placeholder="输入问题后按回车...",
                        scale=4,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        # ── 事件绑定 ──────────────────────────────────
        upload_btn.click(
            do_upload,
            [upload_file],
            [status_text, doc_display, file_picker],
        )
        # 导入后同时刷新文件列表
        import_btn.click(
            do_import,
            [file_picker],
            [status_text, doc_display, file_picker],
        ).then(
            lambda: (gr.Dropdown(choices=scan_for_import()),),
            outputs=[file_picker],
        )

        # 对话
        send_btn.click(do_chat, [msg, chatbot], [chatbot, msg])
        msg.submit(do_chat, [msg, chatbot], [chatbot, msg])

        # 首次加载时刷新
        demo.load(refresh_all, outputs=[doc_display, file_picker, file_picker])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
