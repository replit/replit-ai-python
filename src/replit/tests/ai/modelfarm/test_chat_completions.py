from collections import Counter
from typing import Any, Dict, List

import pytest
from replit.ai.modelfarm import AsyncModelfarm, Modelfarm
from replit.ai.modelfarm.exceptions import BadRequestException
from replit.ai.modelfarm.structs.chat import ChatCompletionMessageRequestParam

# module level constants

MODEL = "chat-bison"

MESSAGES: List[ChatCompletionMessageRequestParam] = [
    {
        "role": "USER",
        "content": "What is the meaning of life?",
    },
]

# kwargs for different endpoints and cases

VALID_KWARGS = {
    "top_p": 0.1,
    "stop": ["\n"],
    "n": 3,
    "provider_extra_parameters": {
        "top_k": 20,
    }
}

INVALID_KWARGS: Dict[str, Any] = {
    "invalid_parameter": 0.5,
}

# stream_chat endpoint does not support the candidateCount arg
VALID_GEN_STREAM_KWARGS = {
    "max_tokens": 128,
    "temperature": 0,
    "top_p": 0.1,
    "provider_extra_parameters": {
        "top_k": 20,
    },
}


def test_chat_model_chat(client: Modelfarm) -> None:
    response = client.chat.completions.create(
        messages=MESSAGES,
        model=MODEL,
        **VALID_KWARGS,
    )

    assert len(response.choices) >= 1

    choice = response.choices[0]

    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_no_kwargs(client: Modelfarm) -> None:
    response = client.chat.completions.create(messages=MESSAGES, model=MODEL)

    assert len(response.choices) == 1

    choice = response.choices[0]

    assert choice.message.content is not None
    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata is not None
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_chat_model_chat_invalid_parameter(client: Modelfarm) -> None:
    with pytest.raises(BadRequestException):
        client.chat.completions.create(
            messages=MESSAGES,
            model=MODEL,
            **INVALID_KWARGS,
        )


@pytest.mark.asyncio
async def test_chat_model_async_chat(async_client: AsyncModelfarm) -> None:
    response = await async_client.chat.completions.create(
        messages=MESSAGES,
        model=MODEL,
        **VALID_KWARGS,
    )

    assert len(response.choices) >= 1

    choice = response.choices[0]

    assert len(choice.message.content) > 10

    choice_metadata = choice.metadata
    assert choice_metadata["safetyAttributes"]["blocked"] is False


@pytest.mark.asyncio
async def test_chat_model_async_chat_invalid_parameter(
        async_client: AsyncModelfarm) -> None:
    with pytest.raises(BadRequestException):
        await async_client.chat.completions.create(messages=MESSAGES,
                                                   model=MODEL,
                                                   **INVALID_KWARGS)


def test_chat_model_stream_chat(client: Modelfarm) -> None:
    responses = list(
        client.chat.completions.create(messages=MESSAGES,
                                       model=MODEL,
                                       stream=True,
                                       **VALID_GEN_STREAM_KWARGS))

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.delta.content is not None
        assert len(choice.delta.content) >= 1


def test_chat_model_stream_chat_invalid_parameter(client: Modelfarm) -> None:
    with pytest.raises(BadRequestException):
        list(
            client.chat.completions.create(messages=MESSAGES,
                                           model=MODEL,
                                           stream=True,
                                           **INVALID_KWARGS))


def test_chat_model_stream_chat_raises_with_choice_count_param(
        client: Modelfarm) -> None:
    """
    Test that stream_chat raises an exception if choice_count is specified.
    """
    with pytest.raises(BadRequestException):
        list(
            client.chat.completions.create(messages=MESSAGES,
                                           model=MODEL,
                                           stream=True,
                                           n=5))


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat(
        async_client: AsyncModelfarm) -> None:
    responses = [
        res async for res in await async_client.chat.completions.create(
            messages=MESSAGES,
            model=MODEL,
            stream=True,
            **VALID_GEN_STREAM_KWARGS)
    ]

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert choice.delta.content is not None
        assert len(choice.delta.content) >= 1


@pytest.mark.asyncio
async def test_chat_model_async_stream_chat_invalid_parameter(
        async_client: AsyncModelfarm) -> None:
    with pytest.raises(BadRequestException):
        async for _ in await async_client.chat.completions.create(
                messages=MESSAGES,
                model=MODEL,
                stream=True,
                **INVALID_KWARGS,
        ):
            pass


def test_chat_model_stream_chat_no_duplicates(client: Modelfarm) -> None:
    # synchronous streaming call
    responses = client.chat.completions.create(
        messages=MESSAGES,
        model=MODEL,
        stream=True,
    )
    counter = Counter()
    for response in responses:
        counter[response.choices[0].delta.content] += 1

    for content, count in counter.items():
        if count > 1:
            print(counter, content)
        assert count == 1
