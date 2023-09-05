import pytest
from replit.ai import EmbedModel
from replit.ai.exceptions import BadRequestException

# module level constants
CONTENT = [{"content": "1 + 1 = "}]


# fixture for creating CompletionModel
@pytest.fixture
def model():
  return EmbedModel("textembedding-gecko")


def test_embed_model_predict(model):
  response = model.embed(CONTENT, {})
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
  pass


@pytest.mark.asyncio
async def test_embed_model_apredict(model):
  response = await model.aembed(CONTENT, {})
  assert len(response.embeddings) == 1

  embedding = response.embeddings[0]

  assert embedding.truncated is False
  assert len(embedding.values) == 768

  choice_metadata = embedding.tokenCountMetadata
  assert embedding.tokenCountMetadata.unbilledTokens == 4
