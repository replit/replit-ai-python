import pytest
from replit.ai.modelfarm import (
    ImageGenerationModel,
    ImageGenerationModelResponse,
    ImageGenerationModelImageResult,
)
from replit.ai.modelfarm.exceptions import InvalidResponseException

PROMPT = "A donkey walking on clouds"
TEST_LORA_URL = "https://110602490-testlora.gateway.alpha.fal.ai"
FAL_KEY_ID = "dummy-key-id"
FAL_KEY_SECRET = "dummysecret"
INVALID_MODEL_ARCH = "invalid"


# fixture for creating ImageGenerationModel
@pytest.fixture
def model():
    return ImageGenerationModel(
        model_url=TEST_LORA_URL, fal_key_id=FAL_KEY_ID, fal_key_secret=FAL_KEY_SECRET
    )


def test_generate_sync(model):
    response = model.generate(PROMPT)

    assert isinstance(response, ImageGenerationModelResponse)

    assert response.images

    image = response.images[0]

    assert isinstance(image, ImageGenerationModelImageResult)

    assert image.width == 500


def test_generate_sync_invalid_parameter(model):
    with pytest.raises(InvalidResponseException):
        model.generate(prompt=PROMPT, model_architecture=INVALID_MODEL_ARCH)


@pytest.mark.asyncio
async def test_generate_async(model):
    response = await model.async_generate(prompt=PROMPT)

    assert isinstance(response, ImageGenerationModelResponse)

    assert response.images

    image = response.images[0]

    assert isinstance(image, ImageGenerationModelImageResult)

    assert image.width == 500


@pytest.mark.asyncio
async def test_generate_async_invalid_parameter(model):
    with pytest.raises(InvalidResponseException):
        await model.generate(prompt=PROMPT, model_architecture=INVALID_MODEL_ARCH)
