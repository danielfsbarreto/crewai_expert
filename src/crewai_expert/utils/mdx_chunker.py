import re

import tiktoken


class MdxChunker:
    def __init__(self, file_content: str, max_tokens: int = 8192):
        self.file_content = file_content
        self.chunks = []
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

    def chunk(self):
        frontmatter = self._chunk_frontmatter()
        body = self._chunk_body()
        sections = self._chunk_sections(body)
        self.chunks = []

        if frontmatter:
            if self._count_tokens(frontmatter) <= self.max_tokens:
                self.chunks.append(frontmatter)
            else:
                self.chunks.extend(self._split_large_chunk(frontmatter))

        for section in sections:
            if self._count_tokens(section) <= self.max_tokens:
                self.chunks.append(section)
            else:
                self.chunks.extend(self._split_large_chunk(section))

        return self.chunks

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_large_chunk(self, text: str):
        chunks = []
        lines = text.splitlines(keepends=True)
        current_chunk = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._count_tokens(line)

            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks.append("".join(current_chunk).strip())
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        if current_chunk:
            chunks.append("".join(current_chunk).strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _chunk_frontmatter(self):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", self.file_content, re.DOTALL)
        return match.group(0) if match else None

    def _chunk_body(self):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", self.file_content, re.DOTALL)
        return self.file_content[match.end() :] if match else self.file_content

    def _chunk_sections(self, body):
        lines = body.splitlines(keepends=True)
        sections = []
        current_section = []
        inside_code_block = False
        code_block_delim = None
        heading_pattern = re.compile(r"^#+ ")

        for line in lines:
            code_block_start = re.match(r"^(```|~~~)", line)
            if code_block_start:
                delim = code_block_start.group(1)
                if not inside_code_block:
                    inside_code_block = True
                    code_block_delim = delim
                elif delim == code_block_delim:
                    inside_code_block = False
                    code_block_delim = None
                current_section.append(line)
                continue

            if not inside_code_block and heading_pattern.match(line):
                if current_section:
                    sections.append("".join(current_section).strip())
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("".join(current_section).strip())
        return [s for s in sections if s]
