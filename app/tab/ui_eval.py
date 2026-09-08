from typing import Any

import gradio as gr

from component import LlmEval
from util.network import find_available_port


RESULT_HEADERS = [
    "Model",
    "Dataset",
    "Concurrency",
    "Prompts",
    "TTFT (ms)",
    "TPOT (ms)",
    "Throughput/GPU (token/s)",
    "SLA",
]


def create_ui_eval(llm_eval: LlmEval) -> None:
    models = llm_eval.get_evaluation_models()
    evaluation_cases = llm_eval.get_evaluation_cases()

    with gr.Row():
        model = gr.Dropdown(
            choices=models,
            value=models[0],
            label="Model",
            interactive=True,
        )
        evaluation_case = gr.Dropdown(
            choices=evaluation_cases,
            value=evaluation_cases[0],
            label="Evaluation case",
            interactive=True,
        )

    with gr.Row():
        concurrency = gr.Number(
            value=4,
            precision=0,
            label="Concurrency (4, 8, ..., 64, or 170)",
        )
        data_num = gr.Number(
            value=16,
            precision=0,
            label="Prompts (concurrency × 4)",
            interactive=False,
        )

    with gr.Row():
        run_button = gr.Button("Run", variant="primary")
        run_all_button = gr.Button("Run All (4–64)")
        auto_run_button = gr.Button("Auto Run", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")
        generate_report_button = gr.Button("Generate Report")

    status = gr.Textbox(label="Status", value="Stopped", interactive=False)
    auto_status = gr.Textbox(
        label="Auto Run Status",
        value="Stopped",
        interactive=False,
    )
    auto_log = gr.Textbox(label="Auto Run Log", lines=12, interactive=False)
    report_status = gr.Textbox(label="Report", interactive=False)
    log_output = gr.Textbox(label="AISBench log", lines=20, interactive=False)
    results = gr.Dataframe(
        headers=RESULT_HEADERS,
        value=[],
        interactive=False,
        label="Results",
    )
    monitor_timer = gr.Timer(value=2, active=True)

    def calculate_data_num(value: int | float) -> int | None:
        try:
            calculated_data_num = llm_eval.get_evaluation_data_num(value)
        except (TypeError, ValueError):
            calculated_data_num = None
        return calculated_data_num

    def run_one(
        selected_model: str,
        selected_case: str,
        selected_concurrency: int | float,
    ) -> tuple[str, str, list[list[Any]]]:
        evaluation_status = llm_eval.start_evaluation(
            selected_model,
            selected_case,
            selected_concurrency,
        )
        _, evaluation_log, evaluation_results = llm_eval.monitor_evaluation()
        state = (evaluation_status, evaluation_log, evaluation_results)
        return state

    def run_all(
        selected_model: str,
        selected_case: str,
    ) -> tuple[str, str, list[list[Any]]]:
        evaluation_status = llm_eval.start_all_evaluations(
            selected_model,
            selected_case,
        )
        _, evaluation_log, evaluation_results = llm_eval.monitor_evaluation()
        state = (evaluation_status, evaluation_log, evaluation_results)
        return state

    def stop() -> tuple[str, str, list[list[Any]], str, str]:
        llm_eval.stop_all()
        _, evaluation_log, evaluation_results = llm_eval.monitor_evaluation()
        auto_run_status, auto_run_log = llm_eval.monitor_auto_run()
        evaluation_status = "Stopping."
        state = (
            evaluation_status,
            evaluation_log,
            evaluation_results,
            auto_run_status,
            auto_run_log,
        )
        return state

    def monitor() -> tuple[str, str, list[list[Any]], str, str]:
        evaluation_state = llm_eval.monitor_evaluation()
        auto_run_state = llm_eval.monitor_auto_run()
        state = (*evaluation_state, *auto_run_state)
        return state

    concurrency.change(
        fn=calculate_data_num,
        inputs=concurrency,
        outputs=data_num,
    )
    run_button.click(
        fn=run_one,
        inputs=[model, evaluation_case, concurrency],
        outputs=[status, log_output, results],
    )
    run_all_button.click(
        fn=run_all,
        inputs=[model, evaluation_case],
        outputs=[status, log_output, results],
    )
    auto_run_button.click(
        fn=llm_eval.start_auto_run,
        outputs=auto_status,
    )
    stop_button.click(
        fn=stop,
        outputs=[status, log_output, results, auto_status, auto_log],
    )
    generate_report_button.click(
        fn=llm_eval.generate_report,
        outputs=report_status,
    )
    monitor_timer.tick(
        fn=monitor,
        outputs=[status, log_output, results, auto_status, auto_log],
        queue=False,
    )


def build_ui() -> gr.Blocks:
    llm_eval = LlmEval()
    with gr.Blocks(title="Model Evaluation") as ui:
        gr.Markdown("# Model Evaluation")
        create_ui_eval(llm_eval)
    return ui


def main() -> None:
    ui = build_ui()
    server_port = find_available_port(7861)
    ui.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    main()
