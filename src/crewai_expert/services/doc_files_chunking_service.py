import asyncio
import os
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.crewai_expert.clients import GithubClient
from src.crewai_expert.types import DocFile, DocFileChunk
from src.crewai_expert.utils import MdxChunker


class DocFilesChunkingService:
    _QDRANT_URL = os.getenv("QDRANT_URL")
    _QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    _QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX")
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self):
        self._github_client = GithubClient()
        self._qdrant_client = QdrantClient(
            url=self._QDRANT_URL, api_key=self._QDRANT_API_KEY
        )
        self._files = []

        if (
            not self._QDRANT_URL
            or not self._QDRANT_API_KEY
            or not self._QDRANT_COLLECTION_PREFIX
        ):
            raise ValueError(
                "Qdrant URL,API key, and collection prefix must be provided either as environment variables QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION_PREFIX"
            )

        if not self._OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable"
            )
        self._openai_client = OpenAI(api_key=self._OPENAI_API_KEY)
        self._collection_name = self._latest_collection_name()

    @property
    def collection_name(self):
        return self._collection_name

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
            file_chunks = MdxChunker(file_content).chunk_content()

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
        self._create_collection()

        all_chunks = []
        embeddings = []
        points = []
        batch_size = 32

        for file in self._files:
            for chunk in file.chunks:
                all_chunks.append({"text": chunk.text, "metadata": chunk.metadata})

        texts_to_embed = [chunk["text"] for chunk in all_chunks]

        with tqdm(
            total=len(texts_to_embed), desc="Generating embeddings", unit="chunk"
        ) as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch_chunks = texts_to_embed[i : i + batch_size]
                response = self._openai_client.embeddings.create(
                    model="text-embedding-3-large", input=batch_chunks
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch_chunks))

        for i, (chunk_dict, embedding) in enumerate(zip(all_chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=list(embedding),
                payload={
                    "text": chunk_dict["text"],
                    "metadata": chunk_dict["metadata"],
                },
            )
            points.append(point)

        with tqdm(
            total=len(points), desc="Saving embeddings to Qdrant", unit="chunk"
        ) as pbar:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self._qdrant_client.upsert(
                    collection_name=self._collection_name, points=batch
                )
                pbar.update(len(batch))

        print(
            f"Successfully saved {len(points)} embeddings to Qdrant collection '{self._collection_name}'"
        )
        self._delete_collection()

    def _latest_collection_name(self):
        collections = self._qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return collection_names[-1] if collection_names else None

    def _create_collection(self):
        try:
            self._collection_name = f"{self._QDRANT_COLLECTION_PREFIX}-{uuid.uuid4()}"

            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=3072,
                    distance=Distance.COSINE,
                ),
            )

            self._qdrant_client.create_payload_index(
                collection_name=self._collection_name,
                field_name="text",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            self._qdrant_client.create_payload_index(
                collection_name=self._collection_name,
                field_name="metadata.file_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            self._qdrant_client.create_payload_index(
                collection_name=self._collection_name,
                field_name="metadata.order",
                field_schema=PayloadSchemaType.INTEGER,
            )

            print(f"Created Qdrant collection '{self._collection_name}'")
            self._collection_name = self._collection_name
            return self._collection_name

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def _delete_collection(self):
        new_collection = self._qdrant_client.get_collection(self._collection_name)
        if new_collection.points_count == 0:
            self._qdrant_client.delete_collection(self._collection_name)
            print(f"Deleted Qdrant collection '{self._collection_name}'")
            return

        for collection in self._qdrant_client.get_collections().collections:
            if (
                collection.name != self._collection_name
                and self._QDRANT_COLLECTION_PREFIX in collection.name
            ):
                self._qdrant_client.delete_collection(collection.name)
                print(f"Deleted Qdrant collection '{collection.name}'")
