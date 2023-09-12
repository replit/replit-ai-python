from replit.ai.modelfarm.google.language_models import TextEmbeddingModel
import pytest


def test_text_embedding_model_get_embeddings():
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(["What is life?"])
    for embedding in embeddings:
        assert len(embedding.values) == 768
        assert embedding.statistics.truncated is False
        assert embedding.statistics.token_count == 4


@pytest.mark.asyncio
async def test_text_embedding_model_async_get_embeddings():
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = await model.async_get_embeddings(["What is life?"])
    for embedding in embeddings:
        assert len(embedding.values) == 768
        assert embedding.statistics.truncated is False
        assert embedding.statistics.token_count == 4
