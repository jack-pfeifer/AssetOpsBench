import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from evaluation.evaluator import evaluate_answer_details, load_questions
from evaluation.leaderboard import compute_leaderboard
from mcp.server.fastmcp import FastMCP
from mcp_ui_server import create_ui_resource
from mcp_ui_server.core import UIResource
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ui-mcp-server")

mcp = FastMCP(
    "ui",
    instructions=(
        "UI-facing AssetOpsBench tools for listing verified models and asking "
        "natural-language questions through the existing plan-execute workflow."
    ),
)

AVAILABLE_MODELS = [
    "watsonx/ibm/granite-4-h-small",
    "watsonx/mistral-large-2512",
    "watsonx/mistralai/mistral-medium-2505",
    "watsonx/mistralai/mistral-small-3-1-24b-instruct-2503",
    "watsonx/openai/gpt-oss-120b",
]

ASK_UI_PATH = Path(__file__).parent / "assets" / "ask.html"


class ModelInfo(BaseModel):
    id: str
    label: str
    verified: bool = True


class ModelsResult(BaseModel):
    models: list[ModelInfo]


class EvaluationQuestionInfo(BaseModel):
    id: str
    question: str
    evaluation_type: str
    expected_output: str = ""


class EvaluationQuestionsResult(BaseModel):
    questions: list[EvaluationQuestionInfo]


class AskResult(BaseModel):
    question: str
    model_id: str
    success: bool
    answer: str
    latency: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    plan: Optional[list[dict[str, Any]]] = None
    trajectory: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None


class EvaluationResultRow(BaseModel):
    question_id: str
    question: str
    model: str
    mode: str
    success: bool
    score: float
    scored: bool
    latency: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    evaluation_summary: str
    answer: str
    error: Optional[str] = None
    evaluation_details: dict[str, Any] = Field(default_factory=dict)
    plan: Optional[list[dict[str, Any]]] = None
    trajectory: Optional[list[dict[str, Any]]] = None


class LeaderboardRow(BaseModel):
    model: str
    total_questions: int
    average_evaluator_score: float
    average_latency: float
    average_total_tokens: float
    quality_weighted_score_per_1k_tokens: float
    question_scores: dict[str, float] = Field(default_factory=dict)


class EvaluationRunResult(BaseModel):
    success: bool
    leaderboard: list[LeaderboardRow]
    results: list[EvaluationResultRow]
    error: Optional[str] = None
    warning: Optional[str] = None


class EvaluationProgressResult(BaseModel):
    active: bool = False
    status: str = "idle"
    run_id: Optional[str] = None
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    current_question_id: Optional[str] = None
    current_question_number: int = 0
    total_questions: int = 0
    current_model: Optional[str] = None
    elapsed_seconds: float = 0.0
    average_run_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    percent_complete: float = 0.0
    last_completed: Optional[str] = None


EVALUATION_PROGRESS: dict[str, Any] = EvaluationProgressResult().model_dump()


def model_label(model_id: str) -> str:
    label = model_id.removeprefix("watsonx/")
    label = label.replace("/", " / ").replace("-", " ")
    return " ".join(word.capitalize() for word in label.split())


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3)) if text.strip() else 0


def plan_execute_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    return env


def summarize_process_error(error: subprocess.CalledProcessError) -> str:
    output = "\n".join(
        part for part in (error.stderr, error.stdout, str(error)) if part
    )
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ignored_fragments = (
        "Give Feedback / Get Help",
        "LiteLLM.Info",
        "During handling of the above exception",
        "The above exception was the direct cause",
    )
    useful_lines = [
        line
        for line in lines
        if not any(fragment in line for fragment in ignored_fragments)
    ]
    priority_fragments = (
        "litellm.exceptions.",
        "APIConnectionError",
        "AuthenticationError",
        "BadRequestError",
        "RateLimitError",
        "ConnectError",
        "Temporary failure in name resolution",
        "missing environment variable",
        "error:",
    )
    priority_lines = [
        line
        for line in useful_lines
        if any(fragment in line for fragment in priority_fragments)
    ]
    selected = priority_lines[-1:] or useful_lines[-3:] or [str(error)]
    return " | ".join(selected)


def extract_json_output(output: str) -> dict[str, Any]:
    output = output.strip()
    if not output:
        return {}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(output[start : end + 1])


def normalized_question_text(question: str) -> str:
    return " ".join(question.lower().split())


def get_evaluation_questions() -> list[dict[str, Any]]:
    return load_questions()


def question_lookup() -> dict[str, dict[str, Any]]:
    return {
        normalized_question_text(question["question"]): question
        for question in get_evaluation_questions()
    }


def build_question_data(question_text: str, index: int) -> tuple[dict[str, Any], bool]:
    configured_question = question_lookup().get(normalized_question_text(question_text))
    if configured_question:
        return configured_question, True

    return {
        "id": f"custom-{index}",
        "question": question_text,
        "evaluation_type": "custom",
        "expected_output": "No configured evaluator metadata is available for this question.",
    }, False


def leaderboard_rows(summary: dict[str, dict[str, Any]]) -> list[LeaderboardRow]:
    rows = []
    for model, stats in summary.items():
        average_total_tokens = stats["average_total_tokens"]
        average_score = stats["accuracy"]
        quality_weighted_score_per_1k_tokens = (
            (average_score * average_score) / (average_total_tokens / 1000)
            if average_total_tokens
            else 0.0
        )
        rows.append(
            LeaderboardRow(
                model=model,
                total_questions=stats["total_questions"],
                average_evaluator_score=average_score,
                average_latency=stats["average_latency"],
                average_total_tokens=average_total_tokens,
                quality_weighted_score_per_1k_tokens=quality_weighted_score_per_1k_tokens,
            )
        )
    return sorted(rows, key=lambda row: row.average_evaluator_score, reverse=True)


def add_question_scores(
    rows: list[LeaderboardRow],
    results: list[EvaluationResultRow],
) -> list[LeaderboardRow]:
    scores_by_model: dict[str, dict[str, float]] = {}

    for result in results:
        if not result.scored:
            continue
        scores_by_model.setdefault(result.model, {})[result.question_id] = result.score

    for row in rows:
        row.question_scores = scores_by_model.get(row.model, {})

    return rows


def update_evaluation_progress(**updates: Any) -> None:
    EVALUATION_PROGRESS.update(updates)


def progress_snapshot() -> EvaluationProgressResult:
    started_at = EVALUATION_PROGRESS.get("started_at")
    completed_runs = int(EVALUATION_PROGRESS.get("completed_runs", 0))
    total_runs = int(EVALUATION_PROGRESS.get("total_runs", 0))
    elapsed_seconds = time.time() - started_at if started_at else 0.0
    average_run_seconds = elapsed_seconds / completed_runs if completed_runs else 0.0
    remaining_runs = max(total_runs - completed_runs, 0)
    estimated_remaining_seconds = (
        average_run_seconds * remaining_runs if completed_runs else None
    )
    percent_complete = completed_runs / total_runs if total_runs else 0.0

    snapshot = dict(EVALUATION_PROGRESS)
    snapshot.pop("started_at", None)
    snapshot.update(
        elapsed_seconds=elapsed_seconds,
        average_run_seconds=average_run_seconds,
        estimated_remaining_seconds=estimated_remaining_seconds,
        percent_complete=percent_complete,
    )
    return EvaluationProgressResult(**snapshot)


@mcp.tool(title="List Verified Models")
def list_models() -> ModelsResult:
    """Return the verified model IDs available for the AoB Ask UI."""
    return ModelsResult(
        models=[
            ModelInfo(id=model_id, label=model_label(model_id))
            for model_id in AVAILABLE_MODELS
        ]
    )


@mcp.tool(title="List Evaluation Questions")
def list_evaluation_questions() -> EvaluationQuestionsResult:
    """Return the configured evaluation questions used for scored UI runs."""
    return EvaluationQuestionsResult(
        questions=[
            EvaluationQuestionInfo(
                id=question["id"],
                question=question["question"],
                evaluation_type=question.get("evaluation_type", ""),
                expected_output=question.get("expected_output", ""),
            )
            for question in get_evaluation_questions()
        ]
    )


@mcp.tool(title="Show AssetOpsBench Ask UI")
def show_ask_ui() -> list[UIResource]:
    """Display the AssetOpsBench Ask UI for MCP-UI hosts."""
    html = ASK_UI_PATH.read_text(encoding="utf-8")
    ui_resource = create_ui_resource(
        {
            "uri": "ui://assetopsbench/ask",
            "content": {
                "type": "rawHtml",
                "htmlString": html,
            },
            "encoding": "text",
        }
    )
    ui_resource.resource.mimeType = "text/html;profile=mcp-app"
    return [ui_resource]


@mcp.tool(title="Ask AssetOpsBench")
def ask_aob(
    question: str,
    model_id: str,
    include_plan: bool = True,
    include_trajectory: bool = True,
) -> AskResult:
    """Ask AssetOpsBench a natural-language question with the selected model."""
    if model_id not in AVAILABLE_MODELS:
        return AskResult(
            question=question,
            model_id=model_id,
            success=False,
            answer="",
            latency=0.0,
            estimated_input_tokens=estimate_tokens(question),
            estimated_output_tokens=0,
            estimated_total_tokens=estimate_tokens(question),
            plan=[] if include_plan else None,
            trajectory=[] if include_trajectory else None,
            error=f"Unsupported model_id: {model_id}",
        )

    start_time = time.time()
    try:
        completed = subprocess.run(
            ["uv", "run", "plan-execute", "--model-id", model_id, "--json", question],
            capture_output=True,
            env=plan_execute_environment(),
            text=True,
            check=True,
        )
        latency = time.time() - start_time
        payload = extract_json_output(completed.stdout)
        answer = payload.get("answer", "")
        input_tokens = estimate_tokens(question)
        output_tokens = estimate_tokens(answer)

        return AskResult(
            question=question,
            model_id=model_id,
            success=True,
            answer=answer,
            latency=latency,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_total_tokens=input_tokens + output_tokens,
            plan=payload.get("plan", []) if include_plan else None,
            trajectory=payload.get("trajectory", []) if include_trajectory else None,
            error=None,
        )
    except subprocess.CalledProcessError as error:
        latency = time.time() - start_time
        logger.error("plan-execute failed: %s", error)
        input_tokens = estimate_tokens(question)

        return AskResult(
            question=question,
            model_id=model_id,
            success=False,
            answer="",
            latency=latency,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=0,
            estimated_total_tokens=input_tokens,
            plan=[] if include_plan else None,
            trajectory=[] if include_trajectory else None,
            error=summarize_process_error(error),
        )
    except json.JSONDecodeError as error:
        latency = time.time() - start_time
        input_tokens = estimate_tokens(question)

        return AskResult(
            question=question,
            model_id=model_id,
            success=False,
            answer="",
            latency=latency,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=0,
            estimated_total_tokens=input_tokens,
            plan=[] if include_plan else None,
            trajectory=[] if include_trajectory else None,
            error=f"Could not parse plan-execute JSON output: {error}",
        )


@mcp.tool(title="Get Evaluation Progress")
def get_evaluation_progress() -> EvaluationProgressResult:
    """Return progress for the currently running UI evaluation, if any."""
    return progress_snapshot()


@mcp.tool(title="Run AssetOpsBench Evaluation")
def run_evaluation(
    questions: list[str],
    model_ids: list[str],
) -> EvaluationRunResult:
    """Run selected questions against selected models and return leaderboard results."""
    clean_questions = [question.strip() for question in questions if question.strip()]
    clean_model_ids = [model_id.strip() for model_id in model_ids if model_id.strip()]

    if not clean_questions:
        return EvaluationRunResult(
            success=False,
            leaderboard=[],
            results=[],
            error="Enter at least one question.",
        )

    if not clean_model_ids:
        return EvaluationRunResult(
            success=False,
            leaderboard=[],
            results=[],
            error="Select at least one model.",
        )

    unsupported_models = [
        model_id for model_id in clean_model_ids if model_id not in AVAILABLE_MODELS
    ]
    if unsupported_models:
        return EvaluationRunResult(
            success=False,
            leaderboard=[],
            results=[],
            error=f"Unsupported model_id: {', '.join(unsupported_models)}",
        )

    rows: list[EvaluationResultRow] = []
    scored_rows_for_leaderboard: list[dict[str, Any]] = []
    run_id = uuid.uuid4().hex
    total_runs = len(clean_questions) * len(clean_model_ids)
    update_evaluation_progress(
        active=True,
        status="running",
        run_id=run_id,
        total_runs=total_runs,
        completed_runs=0,
        failed_runs=0,
        current_question_id=None,
        current_question_number=0,
        total_questions=len(clean_questions),
        current_model=None,
        last_completed=None,
        started_at=time.time(),
    )

    for question_index, question_text in enumerate(clean_questions, start=1):
        question_data, scored = build_question_data(question_text, question_index)

        for model_id in clean_model_ids:
            update_evaluation_progress(
                current_question_id=question_data["id"],
                current_question_number=question_index,
                current_model=model_id,
            )
            ask_result = ask_aob(
                question=question_text,
                model_id=model_id,
                include_plan=True,
                include_trajectory=True,
            )

            if not ask_result.success:
                evaluation = {
                    "score": 0.0,
                    "summary": f"run failed: {(ask_result.error or 'unknown error')[:160]}",
                }
                score = 0.0
                evaluation_summary = evaluation["summary"]
            elif scored:
                evaluation = evaluate_answer_details(ask_result.answer, question_data)
                score = evaluation["score"]
                evaluation_summary = evaluation["summary"]
            else:
                evaluation = {
                    "score": 0.0,
                    "summary": "not scored; question does not match configured evaluator metadata",
                }
                score = 0.0
                evaluation_summary = evaluation["summary"]

            row = EvaluationResultRow(
                question_id=question_data["id"],
                question=question_text,
                model=model_id,
                mode="live",
                success=ask_result.success,
                score=score,
                scored=scored,
                latency=ask_result.latency,
                estimated_input_tokens=ask_result.estimated_input_tokens,
                estimated_output_tokens=ask_result.estimated_output_tokens,
                estimated_total_tokens=ask_result.estimated_total_tokens,
                evaluation_summary=evaluation_summary,
                answer=ask_result.answer,
                error=ask_result.error,
                evaluation_details=evaluation,
                plan=ask_result.plan,
                trajectory=ask_result.trajectory,
            )
            rows.append(row)

            if scored:
                scored_rows_for_leaderboard.append(row.model_dump())

            update_evaluation_progress(
                completed_runs=len(rows),
                failed_runs=sum(1 for result_row in rows if not result_row.success),
                last_completed=f"{question_data['id']} on {model_label(model_id)}",
            )

    warning = None
    if any(not row.scored for row in rows):
        warning = (
            "Some questions were run but not scored because they do not match "
            "the configured evaluation questions."
        )

    summary = compute_leaderboard(scored_rows_for_leaderboard)
    leaderboard = add_question_scores(
        leaderboard_rows(summary),
        rows,
    )
    result = EvaluationRunResult(
        success=True,
        leaderboard=leaderboard,
        results=rows,
        warning=warning,
    )
    update_evaluation_progress(
        active=False,
        status="complete",
        current_question_id=None,
        current_model=None,
    )
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
