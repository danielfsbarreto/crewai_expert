from typing import Literal

from pydantic import BaseModel


class CrewaiExpertState(BaseModel):
    # Inputs
    prompt: str | None = None
    run_type: Literal["update_embeddings", "answer_prompt"] = "answer_prompt"

    # State data filled by the agents
    final_answer: str | None = None
