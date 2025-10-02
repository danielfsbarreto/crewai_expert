from typing import List, Literal, Optional

from pydantic import BaseModel

from src.crewai_expert.types.doc_file import DocFile


class CrewaiExpertState(BaseModel):
    # Inputs
    prompt: Optional[str] = None
    run_type: Literal["update_embeddings", "answer_prompt"] = "answer_prompt"

    # State data filled by the agents
    doc_files: List[DocFile] = []
    final_answer: Optional[str] = None
