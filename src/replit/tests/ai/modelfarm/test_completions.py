from typing import Any, Dict

import pytest
from replit.ai.modelfarm import AsyncModelfarm, Modelfarm
from replit.ai.modelfarm.exceptions import BadRequestException

# module level constants
PROMPT = ["1 + 1 = "]
LONG_PROMPT = [
    "A very long answer to the question of what is the meaning of life is "
]
VALID_KWARGS = {
    "top_p": 0.1,
    "stop": ["\n"],
    "n": 5,
    "provider_extra_parameters": {
        "top_k": 20,
    },
}
# stream_complete endpoint does not support the candidateCount arg
VALID_GEN_STREAM_KWARGS = {
    "max_tokens": 128,
    "temperature": 0,
    "top_p": 0.1,
    "provider_extra_parameters": {
        "top_k": 20,
    },
}
INVALID_KWARGS: Dict[str, Any] = {
    "invalid_parameter": 0.5,
}

MODEL = "text-bison"


def test_completion_model_complete(client: Modelfarm) -> None:
    response = client.completions.create(
        prompt=PROMPT,
        model=MODEL,
        **VALID_KWARGS,
    )

    assert len(response.choices) >= 1

    choice = response.choices[0]

    assert "2" in choice.text

    choice_metadata = choice.metadata
    assert choice_metadata is not None
    assert choice_metadata["safetyAttributes"]["blocked"] is False


def test_completion_model_complete_invalid_parameter(
        client: Modelfarm) -> None:
    with pytest.raises(BadRequestException):
        client.completions.create(prompt=PROMPT, model=MODEL, **INVALID_KWARGS)


@pytest.mark.asyncio
async def test_completion_model_async_complete(
        async_client: AsyncModelfarm) -> None:
    response = await async_client.completions.create(prompt=PROMPT,
                                                     model=MODEL,
                                                     **VALID_KWARGS)

    assert len(response.choices) >= 1

    choice = response.choices[0]

    assert "2" in choice.text

    choice_metadata = choice.metadata

    assert choice_metadata is not None
    assert choice_metadata["safetyAttributes"]["blocked"] is False


@pytest.mark.asyncio
async def test_completion_model_async_complete_invalid_parameter(
        async_client: AsyncModelfarm) -> None:
    with pytest.raises(BadRequestException):
        await async_client.completions.create(prompt=PROMPT,
                                              model=MODEL,
                                              **INVALID_KWARGS)


def test_completion_model_stream_complete(client: Modelfarm) -> None:
    responses = list(
        client.completions.create(prompt=LONG_PROMPT,
                                  model=MODEL,
                                  stream=True,
                                  **VALID_GEN_STREAM_KWARGS))

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert len(choice.text) > 0


def test_completion_model_stream_complete_invalid_parameter(
        client: Modelfarm) -> None:
    with pytest.raises(BadRequestException):
        list(
            client.completions.create(prompt=PROMPT,
                                      model=MODEL,
                                      stream=True,
                                      **INVALID_KWARGS))


@pytest.mark.asyncio
async def test_completion_model_async_stream_complete(
        async_client: AsyncModelfarm) -> None:
    responses = [
        res async for res in await async_client.completions.create(
            prompt=LONG_PROMPT,
            model=MODEL,
            stream=True,
            **VALID_GEN_STREAM_KWARGS)
    ]

    assert len(responses) > 1
    for response in responses:
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert len(choice.text) > 0


@pytest.mark.asyncio
async def test_completion_model_async_stream_complete_invalid_parameter(
        async_client: AsyncModelfarm) -> None:
    with pytest.raises(BadRequestException):
        async for _ in await async_client.completions.create(
                prompt=LONG_PROMPT, model=MODEL, stream=True,
                **INVALID_KWARGS):
            pass
