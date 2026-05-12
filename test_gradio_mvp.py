"""极简 Gradio 测试：逐层加复杂度，定位卡顿点"""
import gradio as gr

# 第一层：纯 Gradio，无任何外部依赖
def echo(msg, history):
    if not msg.strip():
        return history, ""
    history = history or []
    history.append((msg, f"收到了: {msg}"))
    return history, ""

def test1():
    print("=== 测试1：纯 Gradio，无外部依赖 ===")
    with gr.Blocks(title="极简测试") as demo:
        gr.Markdown("## 极简 Gradio 测试")
        chatbot = gr.Chatbot(height=300)
        msg = gr.Textbox(label="输入", placeholder="输入后回车...")
        msg.submit(echo, [msg, chatbot], [chatbot, msg])

    demo.launch(server_name="127.0.0.1", server_port=7870, share=False)

# 第二层：加上 RAG Pipeline
def test2():
    print("=== 测试2：Gradio + RAG Pipeline ===")
    from rag.pipeline import RAGPipeline

    p = RAGPipeline(use_hybrid=True, use_hyde=False)

    def refresh_files():
        files = p.get_file_list()
        if not files:
            return [], gr.Dropdown(choices=[], value=None, interactive=True)
        choices = [f"{f['filename']} ({f['loaded_at']})" for f in files]
        return [
            [f["filename"], f["size_kb"], f["loaded_at"], f["md5"]]
            for f in files
        ], gr.Dropdown(choices=choices, value=choices[0], interactive=True)

    def chat(message, history):
        if not message.strip():
            return history, ""
        history = history or []
        try:
            result = p.query_with_sources(message.strip())
            answer = result["answer"]
            sources = result.get("sources", [])
            if sources:
                answer += "\n\n**参考来源**\n"
                seen = set()
                for s in sources:
                    src = s.get("source", "")
                    if src and src not in seen:
                        seen.add(src)
                        answer += f"- {src}\n"
        except Exception as e:
            answer = f"错误: {e}"
        history.append((message, answer))
        return history, ""

    with gr.Blocks(title="RAG 测试") as demo:
        gr.Markdown("## RAG Pipeline 测试")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 知识库")
                doc_list = gr.DataFrame(
                    headers=["文件名", "大小(KB)", "载入时间", "MD5"],
                    interactive=False,
                )
                doc_selector = gr.Dropdown(label="文档", choices=[], interactive=True)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=300, label="对话")
                msg = gr.Textbox(label="问题", placeholder="输入问题后回车...")

        demo.load(refresh_files, outputs=[doc_list, doc_selector])
        msg.submit(chat, [msg, chatbot], [chatbot, msg])

    demo.launch(server_name="127.0.0.1", server_port=7871, share=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        test2()
    else:
        test1()
