import pytest
from replit.ai.modelfarm import EmbeddingModel
from replit.ai.modelfarm.exceptions import BadRequestException

# module level constants
CONTENT = [{"content": "1 + 1 = "}]


# fixture for creating CompletionModel
@pytest.fixture
def model():
  return EmbeddingModel("textembedding-gecko")


def test_embed_model_embed(model):
  response = model.embed(CONTENT)
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
  pass


@pytest.mark.asyncio
async def test_embed_model_async_embed(model):
  response = await model.async_embed(CONTENT)
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
