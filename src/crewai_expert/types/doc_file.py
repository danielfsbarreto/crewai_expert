from pydantic import BaseModel

from .doc_file_chunk import DocFileChunk


class DocFile(BaseModel):
    path: str
    content: str | None = None
    chunks: list[DocFileChunk] = []
