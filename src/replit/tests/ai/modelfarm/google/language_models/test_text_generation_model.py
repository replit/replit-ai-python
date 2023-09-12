import pytest
from replit.ai.modelfarm.google.language_models import (
    TextGenerationModel,
    TextGenerationResponse,
)

TEST_PARAMETERS = {
    "temperature": 0.5,  # Temperature controls the degree of randomness in token selection.
    "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
    "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
}


def test_text_generation_model_predict():
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        "Give me ten interview questions for the role of program manager.",
        **TEST_PARAMETERS,
    )
    validate_response(response)


@pytest.mark.asyncio
async def test_text_generation_model_async_predict():
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = await model.async_predict(
        "Give me ten interview questions for the role of program manager.",
        **TEST_PARAMETERS,
    )
    validate_response(response)


def test_text_generation_model_predict_streaming():
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict_streaming(
        "Give me 100 interview questions for the role of program manager.",
        **TEST_PARAMETERS,
    )
    result = list(response)
    assert len(result) > 1
    for x in result:
        validate_response(x)


@pytest.mark.asyncio
async def test_text_generation_model_async_predict_streaming():
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.async_predict_streaming(
        "Give me 100 interview questions for the role of program manager.",
        **TEST_PARAMETERS,
    )
    result = []
    async for x in response:
        result.append(x)
    assert len(result) > 1

    for x in result:
        validate_response(x)


def validate_response(response: TextGenerationResponse):
    assert len(response.text) > 1
    assert response.is_blocked is False
    assert response.safety_attributes is not None
