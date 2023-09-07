import pytest
from replit.ai import EmbedModel
from replit.ai.exceptions import BadRequestException

# module level constants
CONTENT = [{"content": "1 + 1 = "}]


# fixture for creating CompletionModel
@pytest.fixture
def model():
  return EmbedModel("textembedding-gecko")


def test_embed_model_generate(model):
  response = model.generate(CONTENT, {})
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
  pass


@pytest.mark.asyncio
async def test_embed_model_async_generate(model):
  response = await model.async_generate(CONTENT, {})
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
