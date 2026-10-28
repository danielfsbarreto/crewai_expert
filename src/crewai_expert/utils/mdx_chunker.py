import tiktoken
from semchunk import chunk


class MdxChunker:
    def __init__(self, file_content: str):
        self.file_content = file_content
        self.max_tokens = 512
        self.overlap = 0.2

    def chunk_content(self):
        return chunk(
            self.file_content,
            chunk_size=self.max_tokens,
            token_counter=self._token_counter,
            overlap=self.overlap,
        )

    def _token_counter(self, text: str) -> int:
        return len(tiktoken.encoding_for_model("text-embedding-3-large").encode(text))
