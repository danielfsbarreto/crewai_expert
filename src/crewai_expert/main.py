#!/usr/bin/env python
from typing import Literal, Optional

from crewai.flow import Flow, listen, router, start
from pydantic import BaseModel

from src.crewai_expert.crews import AnswerCrewaiPromptCrew


class CrewaiExpertState(BaseModel):
    # Inputs
    prompt: Optional[str] = None
    run_type: Literal["update_embeddings", "answer_prompt"] = "answer_prompt"

    # State data filled by the agents
    final_answer: Optional[str] = None


class CrewaiExpertFlow(Flow[CrewaiExpertState]):
    @start()
    def validate_inputs(self):
        if self.state.run_type == "answer_prompt" and self.state.prompt is None:
            raise ValueError("The input `prompt` needs to be provided")

    @router(validate_inputs)
    def identify_path(self):
        return self.state.run_type

    @listen("update_embeddings")
    def update_embeddings_path(self):
        print("The path is", self.state.run_type)

    @listen("answer_prompt")
    def answer_prompt_path(self):
        print("The path is", self.state.run_type)

    @listen(answer_prompt_path)
    def come_up_with_curated_answer(self):
        result = (
            AnswerCrewaiPromptCrew()
            .crew()
            .kickoff(inputs={"prompt": self.state.prompt})
        )
        self.state.final_answer = result.raw

        return {"final_answer": self.state.final_answer}


def kickoff():
    CrewaiExpertFlow().kickoff(
        inputs={"prompt": "Como crio uma ferramenta customizada?"}
    )


def plot():
    CrewaiExpertFlow().plot()


if __name__ == "__main__":
    kickoff()
