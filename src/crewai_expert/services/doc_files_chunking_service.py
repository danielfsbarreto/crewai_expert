import asyncio
import os
import time
import uuid

from openai import OpenAI
from pymongo import ASCENDING, MongoClient
from pymongo.operations import SearchIndexModel
from tqdm import tqdm

from crewai_expert.clients import GithubClient
from crewai_expert.types import DocFile, DocFileChunk
from crewai_expert.utils import MdxChunker

_VECTOR_INDEX_NAME = "vector_index"
_VECTOR_INDEX_POLL_INTERVAL = 5
_VECTOR_INDEX_POLL_TIMEOUT = 300


class DocFilesChunkingService:
    _MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
    _MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")
    _MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self):
        self._github_client = GithubClient()
        self._files = []

        if (
            not self._MONGODB_CONNECTION_STRING
            or not self._MONGODB_DATABASE_NAME
            or not self._MONGODB_COLLECTION_NAME
        ):
            raise ValueError(
                "MongoDB connection string, database name, and collection prefix must be provided "
                "as environment variables MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME, "
                "and MONGODB_COLLECTION_NAME"
            )

        if not self._OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable"
            )

        self._mongo_client = MongoClient(self._MONGODB_CONNECTION_STRING)
        self._db = self._mongo_client[self._MONGODB_DATABASE_NAME]
        self._openai_client = OpenAI(api_key=self._OPENAI_API_KEY)
        self._staging_name = f"{self._MONGODB_COLLECTION_NAME}-staging"

    async def call(self) -> int:
        self._get_files()
        await self._chunk_files()
        return await self._generate_embeddings()

    def _get_files(
        self, docs_path: str = "docs/edge", primary_language: str = "en"
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

        batch_size = 32
        with tqdm(total=len(self._files), desc="Chunking files", unit="file") as pbar:
            for i in range(0, len(self._files), batch_size):
                batch = self._files[i : i + batch_size]
                tasks = [chunk_file(file, pbar) for file in batch]
                await asyncio.gather(*tasks)

        return self._files

    async def _generate_embeddings(self) -> int:
        self._create_staging_collection()

        all_chunks = []
        for file in self._files:
            for chunk in file.chunks:
                all_chunks.append({"text": chunk.text, "metadata": chunk.metadata})

        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self._embed_texts(texts)

        documents = [
            {
                "_id": str(uuid.uuid4()),
                "text": chunk_dict["text"],
                "metadata": chunk_dict["metadata"],
                "embedding": list(embedding),
            }
            for chunk_dict, embedding in zip(all_chunks, embeddings)
        ]

        self._insert_documents(documents)
        self._promote_staging_collection()

        return len(documents)

    def _embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = []

        with tqdm(total=len(texts), desc="Generating embeddings", unit="chunk") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._openai_client.embeddings.create(
                    model="text-embedding-3-large", input=batch
                )
                embeddings.extend(data.embedding for data in response.data)
                pbar.update(len(batch))

        return embeddings

    def _insert_documents(self, documents: list[dict], batch_size: int = 32):
        staging = self._db[self._staging_name]

        with tqdm(
            total=len(documents), desc="Saving embeddings to MongoDB", unit="chunk"
        ) as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                staging.insert_many(batch)
                pbar.update(len(batch))

        print(
            f"Successfully saved {len(documents)} embeddings to staging collection '{self._staging_name}'"
        )

    def _create_staging_collection(self):
        try:
            if self._staging_name in self._db.list_collection_names():
                self._db.drop_collection(self._staging_name)

            self._db.create_collection(self._staging_name)
            staging = self._db[self._staging_name]

            staging.create_index([("metadata.file_path", ASCENDING)])
            staging.create_index([("metadata.order", ASCENDING)])

            search_index = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "path": "embedding",
                            "type": "vector",
                            "numDimensions": 3072,
                            "similarity": "cosine",
                        }
                    ]
                },
                name=_VECTOR_INDEX_NAME,
                type="vectorSearch",
            )
            staging.create_search_index(search_index)

            self._wait_for_vector_index(staging)

            print(f"Created staging collection '{self._staging_name}'")

        except Exception as e:
            print(f"Error creating staging collection: {e}")
            raise

    def _wait_for_vector_index(self, collection):
        start = time.time()
        while time.time() - start < _VECTOR_INDEX_POLL_TIMEOUT:
            indexes = list(collection.list_search_indexes(_VECTOR_INDEX_NAME))
            if indexes and indexes[0].get("status") == "READY":
                return
            print(
                f"Waiting for vector search index to be ready "
                f"({int(time.time() - start)}s elapsed)..."
            )
            time.sleep(_VECTOR_INDEX_POLL_INTERVAL)

        raise TimeoutError(
            f"Vector search index did not become ready within {_VECTOR_INDEX_POLL_TIMEOUT}s"
        )

    def _promote_staging_collection(self):
        self._db.drop_collection(self._MONGODB_COLLECTION_NAME)
        self._db[self._staging_name].rename(self._MONGODB_COLLECTION_NAME)
        print(f"Promoted staging to '{self._MONGODB_COLLECTION_NAME}'")
