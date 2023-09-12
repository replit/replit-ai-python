import pytest
from replit.ai.modelfarm import CompletionModel
from replit.ai.modelfarm.exceptions import BadRequestException

# module level constants
PROMPT = ["1 + 1 = "]
LONG_PROMPT = ["A very long answer to the question of what is the meaning of life is "]
VALID_KWARGS = {"topP": 0.1, "topK": 20, "stopSequences": ["\n"], "candidateCount": 5}
# stream_complete endpoint does not support the candidateCount arg
VALID_GEN_STREAM_KWARGS = {
    "max_output_tokens": 128,
    "temperature": 0,
    "topP": 0.1,
    "topK": 20,
}
INVALID_KWARGS = {"invalid_parameter": 0.5}


# fixture for creating CompletionModel
@pytest.fixture
def model():
    return CompletionModel("text-bison")


def test_completion_model_complete(model):
    response = model.complete(PROMPT, **VALID_KWARGS)

    assert len(response.responses) == 1
    assert len(response.responses[0].choices) == 1

    choice = response.responses[0].choices[0]

    assert "2" in choice.content

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_completion_model_complete_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        model.complete(PROMPT, **INVALID_KWARGS)


@pytest.mark.asyncio
async def test_completion_model_async_complete(model):
    response = await model.async_complete(PROMPT, **VALID_KWARGS)

    assert len(response.responses) == 1
    assert len(response.responses[0].choices) == 1

    choice = response.responses[0].choices[0]

    assert "2" in choice.content

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


@pytest.mark.asyncio
async def test_completion_model_async_complete_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        await model.async_complete(PROMPT, **INVALID_KWARGS)


def test_completion_model_stream_complete(model):
    responses = list(model.stream_complete(LONG_PROMPT, **VALID_GEN_STREAM_KWARGS))

    assert len(responses) > 1
    for response in responses:
        assert len(response.responses) == 1
        assert len(response.responses[0].choices) == 1

        choice = response.responses[0].choices[0]

        assert choice.content is not None


def test_completion_model_stream_complete_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        list(model.stream_complete(PROMPT, **INVALID_KWARGS))


@pytest.mark.asyncio
async def test_completion_model_async_stream_complete(model):
    responses = [
        res
        async for res in model.async_stream_complete(
            LONG_PROMPT, **VALID_GEN_STREAM_KWARGS
        )
    ]

    assert len(responses) > 1
    for response in responses:
        assert len(response.responses) == 1
        assert len(response.responses[0].choices) == 1

        choice = response.responses[0].choices[0]

        assert choice.content is not None


@pytest.mark.asyncio
async def test_completion_model_async_stream_complete_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        async for _ in model.async_stream_complete(LONG_PROMPT, **INVALID_KWARGS):
            pass
