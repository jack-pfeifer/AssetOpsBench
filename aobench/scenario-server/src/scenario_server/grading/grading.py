import logging

import mlflow
from mlflow.entities import Feedback as MLFlowFeedback
from mlflow.tracing.assessment import log_assessment
from scenario_server.entities import SubmissionAnswer, SubmissionScore

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


async def grade_responses(grader, data) -> list[SubmissionScore]:
    submission: list[SubmissionAnswer] = [
        SubmissionAnswer(scenario_id=s["scenario_id"], answer=s["answer"])
        for s in data["submission"]
    ]

    tracking_context = data.get("tracking_context", None)
    results = dict()

    if tracking_context:
        logger.info(f"{tracking_context=}")

        experiment_id: str = tracking_context["experiment_id"]
        run_id: str = tracking_context["run_id"]

        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(run_id=run_id):
            results = await grader(submission)

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
        results = await grader(submission)

    return results
