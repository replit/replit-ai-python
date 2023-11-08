import pytest
from replit.ai.modelfarm import EmbeddingModel

# module level constants
CONTENT = ["1 + 1 = "]


# fixture for creating CompletionModel
@pytest.fixture
def model():
    return EmbeddingModel("textembedding-gecko")


def test_embed_model_embed(model):
    response = model.embed(CONTENT)
    assert len(response.data) == 1

    embedding = response.data[0]

    assert len(embedding.embedding) == 768

    assert embedding.metadata["truncated"] is False
    assert embedding.metadata["tokenCountMetadata"]["unbilledTokens"] == 4


@pytest.mark.asyncio
async def test_embed_model_async_embed(model):
    response = await model.async_embed(CONTENT)
    assert len(response.data) == 1

    embedding = response.data[0]

    assert len(embedding.embedding) == 768

    assert embedding.metadata["truncated"] is False
    assert embedding.metadata["tokenCountMetadata"]["unbilledTokens"] == 4
