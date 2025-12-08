import json
import logging

from huggingface_hub import hf_hub_download
from scenario_server.entities import (
    Scenario,
    ScenarioType,
    SubmissionAnswer,
    SubmissionScore,
)
from scenario_server.grading import evaluation_agent
from scenario_server.handlers.scenario_handler import ScenarioHandler

logger: logging.Logger = logging.getLogger("scenario-server")

HUGGINGFACE_REPO = "ibm-research/AssetOpsBench"
HUGGINGFACE_DATA = "data/scenarios/all_utterance.jsonl"


class AOBTSFMScenarios(ScenarioHandler):
    id = "13aab653-66fe-4fe6-84d8-89f1b18eede3"
    title = "Asset Operations Bench - TSFM"
    description = "Human-authored evaluation prompts for industrial asset agents."

    def __init__(self):
        self.scenario_data = dict()
        try:
            cache: str = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=HUGGINGFACE_DATA,
                repo_type="dataset",
            )

            with open(cache, "r") as f:
                scenario_data = [json.loads(line) for line in f]

            for sd in scenario_data:
                if "type" in sd and sd["type"].lower() == "tsfm":
                    self.scenario_data[str(sd["id"])] = sd

        except Exception as e:
            logger.exception(f"failed to init AOBScenarios: {e=}")

    def _grade_answer(self, entry_id, answer) -> SubmissionScore:
        try:
            unwrap = json.loads(answer)

            c = self.scenario_data[entry_id]["characteristic_form"]
            q = self.scenario_data[entry_id]["text"]
            r = unwrap["result"]
            t = unwrap["trace"]

            result, details = evaluation_agent(
                actual=r,
                charactistic=c,
                query=q,
                trace=t,
            )

            return SubmissionScore(
                scenario_id=entry_id,
                correct=result,
                details=details,
            )
        except Exception as e:
            logger.exception(f"failed to grade {entry_id=} : {e=}")
            logger.debug(f"{entry_id=} / {answer=} / {self.scenario_data[entry_id]}")
            return SubmissionScore(
                scenario_id=entry_id,
                correct=False,
                details=[{"error": f"failed to grade scenario id: {entry_id}"}],
            )

    def scenario_type(self) -> ScenarioType:
        return ScenarioType(id=self.id, title=self.title, description=self.description)

    def fetch_scenarios(self) -> list[Scenario]:
        scenarios = []

        for k, v in self.scenario_data.items():
            try:
                metadata = dict()

                if "category" in v:
                    metadata["category"] = v["category"]

                scenarios.append(
                    Scenario(
                        id=str(k),
                        query=v["text"],
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.exception(f"failed to process {k}, {v} : {e=}")

        return scenarios

    async def grade_responses(
        self, submission: list[SubmissionAnswer]
    ) -> list[SubmissionScore]:

        grade = []
        for entry in submission:
            try:
                entry_id: str = entry.scenario_id
            except Exception as e:
                logger.exception(f"missing scenario id: {entry=}")
                continue

            if entry_id not in self.scenario_data:
                grade.append(
                    SubmissionScore(
                        scenario_id=entry_id,
                        correct=False,
                        details=[{"error": f"unknown scenario id: {entry_id}"}],
                    )
                )
                continue

            g: SubmissionScore = self._grade_answer(entry_id, entry.answer)
            grade.append(g)

        return grade


if __name__ == "__main__":
    import asyncio

    aobs = AOBTSFMScenarios()
    submission: list[SubmissionAnswer] = [
        SubmissionAnswer(
            scenario_id="Q.S5",
            answer='[{"scenario_id": "Q.S5.0", "answer": ""}]',
        ),
        SubmissionAnswer(
            scenario_id="201",
            answer="",
        ),
        SubmissionAnswer(
            scenario_id="201",
            answer=json.dumps(
                {
                    "trace": "Fetch csv data. Use 1 percent of data to fine tune model.",
                    "result": ["saved_model_file=/tmp/model"],
                }
            ),
        ),
    ]
    grade: list[SubmissionScore] = asyncio.run(
        aobs.grade_responses(submission=submission)
    )
    print(f"{grade=}")
