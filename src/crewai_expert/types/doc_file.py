from typing import List, Optional

from pydantic import BaseModel

from src.crewai_expert.types.doc_file_chunk import DocFileChunk


class DocFile(BaseModel):
    path: str
    content: Optional[str] = None
    chunks: List[DocFileChunk] = []
