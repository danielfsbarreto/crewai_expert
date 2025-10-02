import asyncio

from tqdm import tqdm

from src.crewai_expert.clients import GithubClient
from src.crewai_expert.types import DocFile, DocFileChunk
from src.crewai_expert.utils import MdxChunker


class DocFilesChunkingService:
    def __init__(self):
        self._github_client = GithubClient()
        self._files = []

    async def call(self):
        self._get_files()
        await self._chunk_files()
        await self._generate_embeddings()

        return self._files

    def _get_files(
        self, docs_path: str = "docs", primary_language: str = "en"
    ) -> list[DocFile]:
        files_paths = self._github_client.get_file_paths(
            f"{docs_path}/{primary_language}"
        )

        self._files = [DocFile(path=path) for path in files_paths]

    async def _chunk_files(self):
        async def chunk_file(file: DocFile, pbar: tqdm):
            file_content = await self._github_client.get_file_content_async(file.path)
            file_chunks = MdxChunker(file_content).chunk()

            file.content = file_content
            file.chunks = [
                DocFileChunk(
                    text=chunk, metadata={"order": idx, "file_path": file.path}
                )
                for idx, chunk in enumerate(file_chunks)
            ]

            pbar.update(1)

        batch_size = 10
        with tqdm(total=len(self._files), desc="Chunking files", unit="file") as pbar:
            for i in range(0, len(self._files), batch_size):
                batch = self._files[i : i + batch_size]
                tasks = [chunk_file(file, pbar) for file in batch]
                await asyncio.gather(*tasks)

        return self._files

    async def _generate_embeddings(self):
        pass
