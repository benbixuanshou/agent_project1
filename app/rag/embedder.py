from langchain_community.embeddings import DashScopeEmbeddings
from app.config import settings


class EmbeddingService:
    def __init__(self):
        self.model = DashScopeEmbeddings(
            model=settings.dashscope_embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts"""
        return await self.model.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text"""
        return await self.model.aembed_query(text)

    def sync_embed_query(self, text: str) -> list[float]:
        """Synchronous version for Milvus search"""
        return self.model.embed_query(text)


def init_embedding_service() -> EmbeddingService:
    return EmbeddingService()
