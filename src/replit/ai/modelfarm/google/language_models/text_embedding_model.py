from dataclasses import dataclass
from replit.ai.modelfarm.embedding_model import EmbeddingModel, EmbeddingModelResponse
from typing import List, Dict, Any


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
    request = self.__ready_input(content)
    # since this model only takes the content param, we don't pass kwargs
    response = self.underlying_model.embed(request)
    return self.__ready_response(response)

  async def async_get_embeddings(self,
                                 content: List[str]) -> List[TextEmbedding]:
    request = self.__ready_input(content)

    # since this model only takes the content param, we don't pass kwargs
    response = await self.underlying_model.async_embed(request)
    return self.__ready_response(response)

  def __ready_input(self, content: List[str]) -> List[Dict[str, Any]]:
    return [{"content": x} for x in content]

  def __ready_response(
      self, response: EmbeddingModelResponse) -> List[TextEmbedding]:

    def transform_response(x):
      metadata = x.tokenCountMetadata
      tokenCount = metadata.unbilledTokens + metadata.billableTokens
      stats = TextEmbeddingStatistics(tokenCount, x.truncated)
      return TextEmbedding(stats, x.values)

    return [transform_response(x) for x in response.embeddings]
