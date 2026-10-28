from pydantic import BaseModel


class DocFileChunk(BaseModel):
    text: str
    metadata: dict
