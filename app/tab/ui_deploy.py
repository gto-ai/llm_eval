import gradio as gr

from component import LlmEval
from util.network import find_available_port


def create_ui_deploy(llm_eval: LlmEval) -> None:
    models = llm_eval.get_models()
    deployment_cases = llm_eval.get_deployment_cases()

    with gr.Row():
        model = gr.Dropdown(
            choices=models,
            value=models[0],
            label="Model",
            interactive=True,
        )
        deployment_case = gr.Dropdown(
            choices=deployment_cases,
            value=deployment_cases[0],
            label="Deployment case",
            interactive=True,
        )

    with gr.Row():
        deploy_button = gr.Button("Deploy", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")

    status = gr.Textbox(label="Status", value="Stopped", interactive=False)
    log_output = gr.Textbox(label="Deployment log", lines=24, interactive=False)
    monitor_timer = gr.Timer(value=2, active=True)

    def deploy_model(selected_model: str, selected_case: str) -> tuple[str, str]:
        deployment_status = llm_eval.start_deployment(selected_model, selected_case)
        deployment_log = ""
        result = (deployment_status, deployment_log)
        return result

    def stop_model() -> tuple[str, str]:
        deployment_status = llm_eval.stop_deployment()
        _, deployment_log = llm_eval.monitor_deployment()
        result = (deployment_status, deployment_log)
        return result

    deploy_button.click(
        fn=deploy_model,
        inputs=[model, deployment_case],
        outputs=[status, log_output],
    )
    stop_button.click(
        fn=stop_model,
        outputs=[status, log_output],
    )
    monitor_timer.tick(
        fn=llm_eval.monitor_deployment,
        outputs=[status, log_output],
        queue=False,
    )


def build_ui() -> gr.Blocks:
    llm_eval = LlmEval()
    with gr.Blocks(title="Model Deployment") as ui:
        gr.Markdown("# Model Deployment")
        create_ui_deploy(llm_eval)
    return ui


def main() -> None:
    ui = build_ui()
    server_port = find_available_port(7860)
    ui.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    main()
