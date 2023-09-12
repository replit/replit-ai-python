import pytest
from replit.ai.modelfarm import ChatModel
from replit.ai.modelfarm.exceptions import BadRequestException
from replit.ai.modelfarm.structs import ChatSession, ChatMessage, ChatExample
from collections import Counter

# module level constants
PROMPT = [
    ChatSession(
        context="You are philosphy bot.",
        examples=[
            ChatExample(
                input=ChatMessage(content="1 + 1"), output=ChatMessage(content="2")
            )
        ],
        messages=[
            ChatMessage(author="USER", content="What is the meaning of life?"),
        ],
    )
]

# kwargs for different endpoints and cases

VALID_KWARGS = {"topP": 0.1, "topK": 20, "stopSequences": ["\n"], "candidateCount": 5}
# stream_chat endpoint does not support the candidateCount arg
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
    return ChatModel("chat-bison")


def test_chat_model_chat(model):
    response = model.chat(PROMPT, **VALID_KWARGS)

    assert len(response.responses) == 1
    assert len(response.responses[0].candidates) == 1

    candidate = response.responses[0].candidates[0]

    assert len(candidate.message.content) > 10

    choice_metadata = candidate.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_no_kwargs(model):
    response = model.chat(PROMPT)

    assert len(response.responses) == 1
    assert len(response.responses[0].candidates) == 1

    candidate = response.responses[0].candidates[0]

    assert len(candidate.message.content) > 10

    choice_metadata = candidate.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        model.chat(PROMPT, **INVALID_KWARGS)


@pytest.mark.asyncio
async def test_chat_model_async_chat(model):
    response = await model.async_chat(PROMPT, **VALID_KWARGS)

    assert len(response.responses) == 1
    assert len(response.responses[0].candidates) == 1

    candidate = response.responses[0].candidates[0]

    assert len(candidate.message.content) > 10

    choice_metadata = candidate.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


@pytest.mark.asyncio
async def test_chat_model_async_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        await model.async_chat(PROMPT, **INVALID_KWARGS)


def test_chat_model_stream_chat(model):
    responses = list(model.stream_chat(PROMPT, **VALID_GEN_STREAM_KWARGS))

    assert len(responses) > 1
    for response in responses:
        assert len(response.responses) == 1
        assert len(response.responses[0].candidates) == 1

        candidate = response.responses[0].candidates[0]
        assert len(candidate.message.content) >= 1


def test_chat_model_stream_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        list(model.stream_chat(PROMPT, **INVALID_KWARGS))


def test_chat_model_stream_chat_raises_with_candidate_count_param(model):
    """
    Test that stream_chat raises an exception if candidate_count is specified.
    """
    INVALID_CANDIDATE_COUNT_KWARGS = {"candidateCount": 5}
    with pytest.raises(BadRequestException):
        list(model.stream_chat(PROMPT, **INVALID_CANDIDATE_COUNT_KWARGS))


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat(model):
    responses = [
        res async for res in model.async_stream_chat(PROMPT, **VALID_GEN_STREAM_KWARGS)
    ]

    assert len(responses) > 1
    for response in responses:
        assert len(response.responses) == 1
        assert len(response.responses[0].candidates) == 1

        candidate = response.responses[0].candidates[0]
        assert len(candidate.message.content) >= 1


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat_invalid_parameter(model):
    with pytest.raises(BadRequestException):
        async for _ in model.async_stream_chat(PROMPT, **INVALID_KWARGS):
            pass


def test_chat_model_chat_no_duplicates(model):
    chat_session: ChatSession = ChatSession(
        context="You are philosophy bot.",
        examples=[
            ChatExample(
                input=ChatMessage(content="1 + 1"), output=ChatMessage(content="2")
            )
        ],
        messages=[
            ChatMessage(author="USER", content="What is the meaning of life?"),
        ],
    )

    # synchronous streaming call
    responses = model.stream_chat([chat_session])
    counter = Counter()
    for response in responses:
        counter[response.responses[0].candidates[0].message.content] += 1

    for content, count in counter.items():
        if count > 1:
            print(counter)
        assert count == 1
