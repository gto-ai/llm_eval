import gradio as gr

from app.tab.ui_deploy import create_ui_deploy
from app.tab.ui_eval import create_ui_eval
from component import LlmEval


def build_ui() -> gr.Blocks:
    llm_eval = LlmEval()

    with gr.Blocks(title="LLM Evaluation") as ui:
        gr.Markdown("# LLM Evaluation")
        with gr.Tab("Deploy"):
            create_ui_deploy(llm_eval)
        with gr.Tab("Evaluate"):
            create_ui_eval(llm_eval)

    return ui


def main() -> None:
    ui = build_ui()
    ui.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
