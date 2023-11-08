from dataclasses import dataclass
from typing import List

from replit.ai.modelfarm.embedding_model import (
    EmbeddingModel,
    EmbeddingModelResponse,
)
from replit.ai.modelfarm.structs import Embedding


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
        self.underlying_model = EmbeddingModel(model_id)

    @staticmethod
    def from_pretrained(model_id: str) -> "TextEmbeddingModel":
        return TextEmbeddingModel(model_id)

    # this model only takes in the content parameter and nothing else
    def get_embeddings(self, content: List[str]) -> List[TextEmbedding]:
        # since this model only takes the content param, we don't pass kwargs
        response = self.underlying_model.embed(content)
        return self.__ready_response(response)

    async def async_get_embeddings(self,
                                   content: List[str]) -> List[TextEmbedding]:
        # since this model only takes the content param, we don't pass kwargs
        response = await self.underlying_model.async_embed(content)
        return self.__ready_response(response)

    def __ready_response(
            self, response: EmbeddingModelResponse) -> List[TextEmbedding]:

        def transform_response(x: Embedding):
            token_metadata = x.metadata["tokenCountMetadata"]
            tokenCount = token_metadata["unbilledTokens"] + token_metadata["billableTokens"]
            stats = TextEmbeddingStatistics(tokenCount, x.metadata["truncated"])
            return TextEmbedding(stats, x.embedding)

        return [transform_response(x) for x in response.data]
