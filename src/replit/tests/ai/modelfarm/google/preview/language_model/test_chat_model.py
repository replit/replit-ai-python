import pytest
from replit.ai.modelfarm.google.preview.language_models import (
    ChatModel,
    InputOutputTextPair,
)
from replit.ai.modelfarm.google.structs import TextGenerationResponse

parameters = {
    # Temperature controls the degree of randomness in token selection.
    "temperature": 0.5,
    # Token limit determines the maximum amount of text output.
    "max_output_tokens": 256,
    # Tokens are selected from most probable to least until the sum of
    # their probabilities equals the top_p value.
    "top_p": 0.95,
    # A top_k of 1 means the selected token is the most probable among all tokens.
    "top_k": 40,
}


def test_chat_model_send_message():
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    chat = chat_model.start_chat(
        context=("My name is Miles. You are an astronomer, "
                 "knowledgeable about the solar system."),
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )

    response = chat.send_message(
        "How many planets are there in the solar system?", **parameters)
    validate_response(response)


@pytest.mark.asyncio
async def test_chat_model_async_send_message():
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    chat = chat_model.start_chat(
        context=("My name is Miles. You are an astronomer, "
                 "knowledgeable about the solar system."),
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )

    response = await chat.async_send_message(
        "How many planets are there in the solar system?", **parameters)
    validate_response(response)


def test_chat_model_send_message_stream():
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    chat = chat_model.start_chat(
        context=("My name is Miles. You are an astronomer, "
                 "knowledgeable about the solar system."),
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )

    responses = list(
        chat.send_message_stream("Name as many different stars as you can.",
                                 **parameters))
    assert len(responses) > 1

    assert any(len(res.text) > 1 for res in responses)
    assert all(response.is_blocked is False for response in responses)
    assert all(response.safety_attributes is not None
               for response in responses)


@pytest.mark.asyncio
async def test_chat_model_async_send_message_stream():
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    chat = chat_model.start_chat(
        context=("My name is Miles. You are an astronomer, "
                 "knowledgeable about the solar system."),
        examples=[
            InputOutputTextPair(
                input_text="How many moons does Mars have?",
                output_text="The planet Mars has two moons, Phobos and Deimos.",
            ),
        ],
    )
    responses = [
        res async for res in chat.async_send_message_stream(
            "Name as many different stars as you can.", **parameters)
    ]

    assert len(responses) > 1

    assert any(len(res.text) > 1 for res in responses)
    assert all(response.is_blocked is False for response in responses)
    assert all(response.safety_attributes is not None
               for response in responses)


def validate_response(response: TextGenerationResponse):
    assert len(response.text) > 1
    assert response.is_blocked is False
    assert response.safety_attributes is not None
