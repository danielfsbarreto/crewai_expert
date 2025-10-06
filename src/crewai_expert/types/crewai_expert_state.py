from typing import Literal, Optional

from pydantic import BaseModel


class CrewaiExpertState(BaseModel):
    # Inputs
    prompt: Optional[str] = None
    run_type: Literal["update_embeddings", "answer_prompt"] = "answer_prompt"

    # State data filled by the agents
    final_answer: Optional[str] = None
