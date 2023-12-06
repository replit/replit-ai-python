from dataclasses import dataclass
from typing import List

from replit.ai.modelfarm import AsyncModelfarm, Modelfarm
from replit.ai.modelfarm.structs.embeddings import Embedding, EmbeddingModelResponse


@dataclass
class TextEmbeddingStatistics:
    token_count: float
    truncated: bool


@dataclass
class TextEmbedding:
    statistics: TextEmbeddingStatistics
    values: List[float]


class TextEmbeddingModel:

    def __init__(self, model_id: str):
        self.underlying_model = model_id
        self._client = Modelfarm()
        self._async_client = AsyncModelfarm()

    @staticmethod
    def from_pretrained(model_id: str) -> "TextEmbeddingModel":
        return TextEmbeddingModel(model_id)

    # this model only takes in the content parameter and nothing else
    def get_embeddings(self, content: List[str]) -> List[TextEmbedding]:
        # since this model only takes the content param, we don't pass kwargs
        response = self._client.embeddings.create(input=content,
                                                  model=self.underlying_model)
        return self.__ready_response(response)

    async def async_get_embeddings(self,
                                   content: List[str]) -> List[TextEmbedding]:
        # since this model only takes the content param, we don't pass kwargs
        response = await self._async_client.embeddings.create(
            input=content, model=self.underlying_model)
        return self.__ready_response(response)

    def __ready_response(
            self, response: EmbeddingModelResponse) -> List[TextEmbedding]:

        def transform_response(x: Embedding) -> TextEmbedding:
            metadata = x.metadata or {}
            token_metadata = metadata.get("tokenCountMetadata", {})
            tokenCount: int = token_metadata.get(
                "unbilledTokens", 0) + token_metadata.get("billableTokens", 0)
            stats = TextEmbeddingStatistics(tokenCount, metadata["truncated"])
            return TextEmbedding(stats, x.embedding)

        return [transform_response(x) for x in response.data]
