import os
import uuid

from app.config import settings
from app.rag.chunker import MarkdownChunker, DocumentChunk
from app.rag.embedder import EmbeddingService
from langchain_milvus import Milvus


class IndexingService:
    """
    File indexing pipeline.
    Ported from Java VectorIndexService:
    1. Read file
    2. Delete old data for same source
    3. Chunk document
    4. Generate embeddings
    5. Insert to Milvus
    """

    def __init__(self, vector_store: Milvus, embedder: EmbeddingService):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = MarkdownChunker(
            max_size=settings.chunk_max_size,
            overlap=settings.chunk_overlap,
        )

    async def index_file(self, file_path: str):
        """Index a single file into Milvus"""
        normalized_path = os.path.normpath(file_path).replace("\\", "/")

        # 1. Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. Delete existing data for this source
        await self._delete_existing(normalized_path)

        # 3. Chunk document
        chunks = self.chunker.chunk(content, normalized_path)

        # 4. Generate embeddings and insert
        for i, chunk in enumerate(chunks):
            vector = await self.embedder.embed_query(chunk.content)

            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{normalized_path}_{chunk.chunk_index}"))

            metadata = {
                "_source": normalized_path,
                "_extension": os.path.splitext(normalized_path)[1],
                "_file_name": os.path.basename(normalized_path),
                "chunk_index": chunk.chunk_index,
                "total_chunks": len(chunks),
                "_cluster": settings.knowledge_cluster_tag,
            }
            if chunk.metadata.get("title"):
                metadata["title"] = chunk.metadata["title"]

            # Insert with dynamic field support
            entity = {"id": chunk_id, "content": chunk.content, "vector": vector}
            entity.update(metadata)
            self.vector_store.col.insert([entity])

        # Flush to ensure data is searchable
        self.vector_store.col.flush()

    async def index_text(self, content: str, source: str = "inline"):
        """Index a text string into Milvus (used for knowledge deposition)."""
        chunks = self.chunker.chunk(content, source)
        for chunk in chunks:
            vector = await self.embedder.embed_query(chunk.content)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}_{chunk.chunk_index}"))
            entity = {
                "id": chunk_id,
                "content": chunk.content,
                "vector": vector,
                "_source": source,
                "_file_name": source,
                "_cluster": settings.knowledge_cluster_tag,
                "chunk_index": chunk.chunk_index,
                "total_chunks": len(chunks),
            }
            self.vector_store.col.insert([entity])
        self.vector_store.col.flush()

    async def _delete_existing(self, source_path: str):
        expr = f'_source == "{source_path}"'
        try:
            self.vector_store.col.delete(expr)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg or "empty" in error_msg:
                return  # 集合为空或不存在，正常情况
            raise  # 连接失败等实际错误，向上抛出
