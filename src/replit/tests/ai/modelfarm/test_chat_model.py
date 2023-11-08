from collections import Counter

import pytest
from replit.ai.modelfarm import ChatModel
from replit.ai.modelfarm.exceptions import BadRequestException

# module level constants

MESSAGES = [
    {
        "role": "USER",
        "content": "What is the meaning of life?",
    }
]

# kwargs for different endpoints and cases

VALID_KWARGS = {"top_p": 0.1, "top_k": 20, "stop": ["\n"], "n": 5}
# stream_chat endpoint does not support the candidateCount arg
VALID_GEN_STREAM_KWARGS = {
    "max_tokens": 128,
    "temperature": 0,
    "top_p": 0.1,
    "top_k": 20,
}
INVALID_KWARGS = {"invalid_parameter": 0.5}


# fixture for creating CompletionModel
@pytest.fixture
def model():
    return ChatModel("chat-bison")


def test_chat_model_chat(model):
    response = model.chat(MESSAGES, **VALID_KWARGS)

    assert len(response.choices) == 1

    choice = response.choices[0]

    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_no_kwargs(model):
    response = model.chat(MESSAGES)

    assert len(response.choices) == 1

    choice = response.choices[0]

    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        model.chat(MESSAGES, **INVALID_KWARGS)


@pytest.mark.asyncio
async def test_chat_model_async_chat(model):
    response = await model.async_chat(MESSAGES, **VALID_KWARGS)

    assert len(response.choices) == 1

    choice = response.choices[0]

    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


@pytest.mark.asyncio
async def test_chat_model_async_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        await model.async_chat(MESSAGES, **INVALID_KWARGS)


def test_chat_model_stream_chat(model):
    responses = list(model.stream_chat(MESSAGES, **VALID_GEN_STREAM_KWARGS))

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1

        choice = response.choices[0]

        choice = response.responses[0].choices[0]
        assert len(choice.message.content) >= 1


def test_chat_model_stream_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        list(model.stream_chat(MESSAGES, **INVALID_KWARGS))


def test_chat_model_stream_chat_raises_with_choice_count_param(model):
    """
    Test that stream_chat raises an exception if choice_count is specified.
    """
    INVALID_CANDIDATE_COUNT_KWARGS = {"n": 5}
    with pytest.raises(BadRequestException):
        list(model.stream_chat(MESSAGES, **INVALID_CANDIDATE_COUNT_KWARGS))


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat(model):
    responses = [
        res async for res in model.async_stream_chat(MESSAGES, **VALID_GEN_STREAM_KWARGS)
    ]

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert len(choice.message.content) >= 1


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        async for _ in model.async_stream_chat(MESSAGES, **INVALID_KWARGS):
            pass


def test_chat_model_chat_no_duplicates(model):
    # synchronous streaming call
    responses = model.stream_chat(MESSAGES)
    counter = Counter()
    for response in responses:
        counter[response.choices[0].message.content] += 1

    for content, count in counter.items():
        if count > 1:
            print(counter, content)
        assert count == 1
