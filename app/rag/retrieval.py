"""Vector store wrapper: pymilvus direct connection with langchain retriever interface.

Why not langchain-milvus:
  langchain-milvus 0.1.x constructor (Milvus()) hangs indefinitely in Docker
  when the collection already exists. pymilvus 2.6.x connects fine directly.
  The workaround: custom MilvusStore that handles connection + collection
  lifecycle manually, with a LangChain-compatible as_retriever() interface.
  If upgrading langchain-milvus to 0.3+, revisit this decision.
"""

import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pymilvus import Collection, connections, utility

from app.config import settings
from app.rag.embedder import EmbeddingService, init_embedding_service


class MilvusStore:
    """Thin wrapper around pymilvus that provides as_retriever() for langchain."""

    def __init__(self, embedder: EmbeddingService):
        self._embedder = embedder
        self._connect()
        self._ensure_collection()

    def _connect(self):
        for attempt in range(5):
            try:
                connections.connect(
                    host=settings.milvus_host,
                    port=settings.milvus_port,
                    timeout=10,
                    alias="default",
                )
                return
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(3)

    def _ensure_collection(self):
        name = settings.milvus_collection
        if utility.has_collection(name):
            self.col = Collection(name)
            self.col.load()
        else:
            from pymilvus import CollectionSchema, DataType, FieldSchema
            fields = [
                FieldSchema("id", DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=settings.milvus_vector_dim),
                FieldSchema("content", DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.col = Collection(name, schema)
            self.col.create_index(
                field_name="vector",
                index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            )
            self.col.load()

    def as_retriever(self, search_kwargs=None) -> BaseRetriever:
        return MilvusRetriever(self, search_kwargs or {})

    def close(self):
        try:
            connections.disconnect("default")
        except Exception:
            pass


class MilvusRetriever(BaseRetriever):
    def __init__(self, store: MilvusStore, search_kwargs: dict):
        super().__init__()
        self._store = store
        self._kwargs = search_kwargs

    def _get_relevant_documents(self, query: str) -> list[Document]:
        vec = self._store._embedder.sync_embed_query(query)
        top_k = self._kwargs.get("k", settings.rag_top_k)
        results = self._store.col.search(
            data=[vec],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "_file_name", "_source", "*"],
        )
        docs = []
        for hits in results:
            for hit in hits:
                content_val = hit.get("content", "") or ""
                file_name = hit.get("_file_name", "") or ""
                docs.append(Document(
                    page_content=str(content_val),
                    metadata={
                        "score": hit.distance,
                        "_file_name": str(file_name),
                    },
                ))
        return docs


_vector_store_instance = None


def init_vector_store(embedder: EmbeddingService = None) -> MilvusStore:
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance

    if embedder is None:
        embedder = init_embedding_service()

    _vector_store_instance = MilvusStore(embedder)
    return _vector_store_instance
