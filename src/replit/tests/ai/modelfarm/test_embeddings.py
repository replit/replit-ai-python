import pytest
from replit.ai.modelfarm import AsyncModelfarm, Modelfarm

# module level constants
CONTENT = ["1 + 1 = "]

MODEL = "textembedding-gecko"


def test_embed_model_embed(client: Modelfarm) -> None:
    response = client.embeddings.create(input=CONTENT, model=MODEL)
    assert len(response.data) == 1

    embedding = response.data[0]

    assert len(embedding.embedding) == 768

    assert embedding.metadata is not None
    assert embedding.metadata["truncated"] is False
    assert embedding.metadata["tokenCountMetadata"]["unbilledTokens"] == 4


@pytest.mark.asyncio
async def test_embed_model_async_embed(async_client: AsyncModelfarm) -> None:
    response = await async_client.embeddings.create(input=CONTENT, model=MODEL)
    assert len(response.data) == 1

    embedding = response.data[0]

    assert len(embedding.embedding) == 768

    assert embedding.metadata is not None
    assert embedding.metadata["truncated"] is False
    assert embedding.metadata["tokenCountMetadata"]["unbilledTokens"] == 4
