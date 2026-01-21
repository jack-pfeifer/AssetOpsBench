import json
import logging
import ssl
from os import environ

import httpx
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)


def set_ssl_context():
    try:
        ca_file = environ.get("SSL_CERT_FILE", None)
        if ca_file is None:
            logger.debug("setting verify ssl to True")
            return True
        elif str(ca_file).lower() in ["f", "false", "n", "no"]:
            logger.debug("setting verify ssl to False")
            return False
        elif str(ca_file).lower() in ["t", "true", "y", "yes"]:
            logger.debug("setting verify ssl to True")
            return True
        else:
            logger.debug(f"setting verify ssl to context {ca_file=}")
            return ssl.create_default_context(cafile=ca_file)
    except Exception as e:
        logger.exception(f"failed to set ssl context {e=}, defaulting to True")
        return True


verify_ssl = set_ssl_context()


def tag_latest_trace(experiment_id, run_id, scenario_id):
    # tag latest trace with scenario_id
    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        run_id=run_id,
        order_by=["timestamp_ms DESC"],
        max_results=1,
    )

    if len(traces) == 1:
        current_trace_id = list(traces.trace_id)[0]
        mlflow.set_trace_tag(
            trace_id=current_trace_id, key="scenario_id", value=scenario_id
        )
    else:
        logger.error(f"found {len(traces)=}, ignoring")


class AOBench:
    def __init__(self, scenario_uri: str, tracking_uri: str = ""):
        self.scenario_uri: str = scenario_uri
        self.tracking_uri: str = tracking_uri

    async def arun(
        self,
        afunc,
        scenario_id,
        run_name: str = "",
        post_process=None,
        tracking_context: dict | None = None,
        **kwargs,
    ):
        try:
            if tracking_context:
                if run_name != "":
                    mlflow.set_tag("mlflow.runName", run_name)

                with mlflow.start_span(name=scenario_id):
                    result = await afunc(**kwargs)

                eid = tracking_context["experiment_id"]
                rid = tracking_context["run_id"]
                tag_latest_trace(experiment_id=eid, run_id=rid, scenario_id=scenario_id)
            else:
                result = await afunc(**kwargs)

            if post_process:
                result = post_process(result)

            logger.debug(f"{result=}")

            answer = {
                "scenario_id": str(scenario_id),
                "answer": result,
            }

        except Exception as e:
            logger.exception(f"aobench.run failed: {e=}")
            raise

        return answer

    def run(
        self,
        func,
        scenario_id,
        run_name: str = "",
        post_process=None,
        tracking_context: dict | None = None,
        **kwargs,
    ):
        try:
            if tracking_context:
                if run_name != "":
                    mlflow.set_tag("mlflow.runName", run_name)

                with mlflow.start_span(name=scenario_id):
                    result = func(**kwargs)

                eid = tracking_context["experiment_id"]
                rid = tracking_context["run_id"]
                tag_latest_trace(experiment_id=eid, run_id=rid, scenario_id=scenario_id)
            else:
                result = func(**kwargs)

            if post_process:
                result = post_process(result)

            logger.debug(f"{result=}")

            answer = {
                "scenario_id": str(scenario_id),
                "answer": result,
            }

        except Exception as e:
            logger.exception(f"aobench.run failed: {e=}")
            raise

        return answer

    def scenario_types(self) -> dict:
        with httpx.Client(verify=verify_ssl) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-types"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(f"{endpoint}")
            r.raise_for_status()

            return r.json()

    def scenario_set(
        self, scenario_set_id: str, tracking: bool
    ) -> tuple[dict, dict | None]:
        with httpx.Client(verify=verify_ssl) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-set/{scenario_set_id}"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(
                f"{endpoint}",
                params={"tracking": tracking},
            )
            r.raise_for_status()

            rsp = r.json()

            scenario_set = {
                "title": rsp["title"],
                "scenarios": rsp["scenarios"],
            }

            if tracking:
                try:
                    tracking_context = rsp["tracking_context"]

                    if (
                        self.tracking_uri != ""
                        and self.tracking_uri != tracking_context["uri"]
                    ):
                        logger.info(f"tracking uri differs: {tracking_context['uri']}")
                        tracking_context["uri"] = self.tracking_uri

                    tracking_uri = tracking_context["uri"]
                    experiment_id = tracking_context["experiment_id"]
                    run_id = tracking_context["run_id"]

                    logger.info(f"{tracking_uri=} / {experiment_id=} / {run_id=}")

                    mlflow.set_tracking_uri(uri=tracking_uri)

                    mlflow.langchain.autolog()
                    mlflow.set_experiment(experiment_id=experiment_id)

                    mlflow.start_run(run_id=run_id)

                    return scenario_set, tracking_context
                except Exception as e:
                    logger.exception(f"failed to init tracking: {e=}")

            return scenario_set, None

    def grade(
        self,
        scenario_set_id: str,
        answers,
        tracking_context: dict | None,
    ) -> dict:
        with httpx.Client(
            verify=verify_ssl,
            timeout=httpx.Timeout(connect=5.0, read=90.0, write=60.0, pool=5.0),
        ) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-set/{scenario_set_id}/grade"
            logger.debug(f"{endpoint=}")

            jsn = {
                "scenario_set_id": scenario_set_id,
                "submission": answers,
            }

            if tracking_context is not None:
                mlflow.end_run()
                jsn["tracking_context"] = {
                    "experiment_id": tracking_context["experiment_id"],
                    "run_id": tracking_context["run_id"],
                }

            r: httpx.Response = client.post(f"{endpoint}", json=jsn)
            r.raise_for_status()

            return r.json()

    def deferred_grading(
        self,
        scenario_set_id: str,
        answers,
        tracking_context: dict | None,
    ) -> dict:
        with httpx.Client(verify=verify_ssl) as client:
            endpoint: str = (
                f"{self.scenario_uri}/scenario-set/{scenario_set_id}/deferred-grading"
            )
            logger.debug(f"{endpoint=}")

            jsn = {
                "scenario_set_id": scenario_set_id,
                "submission": answers,
            }

            if tracking_context is not None:
                mlflow.end_run()
                jsn["tracking_context"] = {
                    "experiment_id": tracking_context["experiment_id"],
                    "run_id": tracking_context["run_id"],
                }

            r: httpx.Response = client.post(f"{endpoint}", json=jsn)
            r.raise_for_status()

            return r.json()

    def deferred_grading_status(self, grading_id) -> dict:
        with httpx.Client(verify=verify_ssl) as client:
            endpoint: str = f"{self.scenario_uri}/deferred-grading/{grading_id}/status"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(endpoint)
            r.raise_for_status()

            return r.json()

    def deferred_grading_result(self, grading_id) -> dict:
        with httpx.Client(verify=verify_ssl) as client:
            endpoint: str = f"{self.scenario_uri}/deferred-grading/{grading_id}/result"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(endpoint)
            r.raise_for_status()

            return r.json()
