import logging
import uuid

import mlflow
from litestar import get, post
from litestar.exceptions import HTTPException
from litestar.openapi.config import OpenAPIConfig
from litestar.status_codes import HTTP_404_NOT_FOUND
from mlflow.entities import Feedback as MLFlowFeedback
from mlflow.tracing.assessment import log_assessment
from scenario_server.entities import ScenarioSet, ScenarioType, SubmissionAnswer

logger: logging.Logger = logging.getLogger("scenario-server")


REGISTERED_SCENARIO_HANDLERS = dict()


def register_scenario_handlers(handlers: list):
    global REGISTERED_SCENARIO_HANDLERS

    for handler in handlers:
        try:
            REGISTERED_SCENARIO_HANDLERS[handler.id] = handler()
        except Exception as e:
            logger.exception(f"failed to load {handler.title=}: {e=}")


TRACKING_URI: str = ""


def set_tracking_uri(tracking_uri: str):
    global TRACKING_URI

    TRACKING_URI = tracking_uri
    mlflow.set_tracking_uri(uri=tracking_uri)


@get("/scenario-types")
async def scenario_types() -> list[ScenarioType]:
    """Get all scenario types"""
    return [rsh.scenario_type() for rsh in REGISTERED_SCENARIO_HANDLERS.values()]


@get("/scenario-set/{scenario_set_id: str}")
async def fetch_scenario(scenario_set_id: str, tracking: bool = False) -> dict:
    if scenario_set_id not in REGISTERED_SCENARIO_HANDLERS.keys():
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"no scenario set {scenario_set_id}",
        )

    title: str = REGISTERED_SCENARIO_HANDLERS[scenario_set_id].title
    scenario_set: ScenarioSet = REGISTERED_SCENARIO_HANDLERS[
        scenario_set_id
    ].fetch_scenarios()

    if tracking and TRACKING_URI:
        logger.info(f"{tracking=} and {TRACKING_URI=}")

        mlflow.set_experiment(experiment_name=f"{title}")
        with mlflow.start_run(run_name=f"{uuid.uuid4()}") as run:
            experiment_id = run.info.experiment_id
            run_id = run.info.run_id

        return {
            "title": title,
            "scenarios": scenario_set,
            "tracking_context": {
                "uri": TRACKING_URI,
                "experiment_id": experiment_id,
                "run_id": run_id,
            },
        }

    return {
        "title": title,
        "scenarios": scenario_set,
    }


@post("/scenario-set/{scenario_set_id: str}/grade")
async def grade_submission(scenario_set_id: str, data: dict) -> dict:
    if scenario_set_id not in REGISTERED_SCENARIO_HANDLERS.keys():
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"no scenario set {scenario_set_id}",
        )

    submission: list[SubmissionAnswer] = [
        SubmissionAnswer(scenario_id=s["scenario_id"], answer=s["answer"])
        for s in data["submission"]
    ]

    results = dict()
    if "tracking_context" in data:
        tracking_context = data["tracking_context"]
        logger.info(f"{tracking_context=}")

        experiment_id: str = tracking_context["experiment_id"]
        run_id: str = tracking_context["run_id"]

        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(run_id=run_id):
            results = await REGISTERED_SCENARIO_HANDLERS[
                scenario_set_id
            ].grade_responses(submission)

            traces = mlflow.search_traces(experiment_ids=[experiment_id], run_id=run_id)
            correct = 0
            for result in results:
                result_id: str = result.scenario_id

                mask = traces["tags"].apply(
                    lambda d: isinstance(d, dict) and d.get("scenario_id") == result_id
                )
                trace_row = traces[mask]

                try:
                    tid = trace_row.iloc[0]["trace_id"]
                    feedback = MLFlowFeedback(name="Correct", value=result.correct)
                    log_assessment(trace_id=tid, assessment=feedback)

                    if result.correct == True:
                        correct += 1
                except Exception as e:
                    logger.exception(f"failed to log result: {e=}")

                for r in result.details:
                    try:
                        tid = trace_row.iloc[0]["trace_id"]
                        if isinstance(r, MLFlowFeedback):
                            log_assessment(trace_id=tid, assessment=r)
                        else:
                            log_assessment(
                                trace_id=tid,
                                assessment=MLFlowFeedback(
                                    name=r["name"],
                                    value=r["value"],
                                ),
                            )
                    except Exception as e:
                        logger.exception(f"failed to log assessment: {e=}")

            mlflow.set_tag("Correct", f"{correct} / {len(results)}")
    else:
        results = await REGISTERED_SCENARIO_HANDLERS[scenario_set_id].grade_responses(
            submission
        )

    return results


OPENAPI_CONFIG = OpenAPIConfig(
    title="Asset Operations Bench",
    description="",
    version="0.0.1",
)

ROUTE_HANDLERS = [scenario_types, fetch_scenario, grade_submission]
