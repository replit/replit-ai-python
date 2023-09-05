import pytest
from replit.ai import ChatModel
from replit.ai.exceptions import BadRequestException
from replit.ai.structs import ChatSession, ChatMessage, ChatExample

# module level constants
PROMPT = [
    ChatSession(context="You are philosphy bot.",
                examples=[
                    ChatExample(input=ChatMessage(content="1 + 1"),
                                output=ChatMessage(content="2"))
                ],
                messages=[
                    ChatMessage(author="USER",
                                content="What is the meaning of life?"),
                ])
]

VALID_PARAMETERS = {"temperature": 0.5}
INVALID_PARAMETERS = {"invalid_parameter": 0.5}


# fixture for creating CompletionModel
@pytest.fixture
def model():
  return ChatModel("chat-bison")


def test_chat_model_predict(model):
  response = model.predict(PROMPT, VALID_PARAMETERS)

  assert len(response.responses) == 1
  assert len(response.responses[0].candidates) == 1

  candidate = response.responses[0].candidates[0]

  assert len(candidate.message.content) > 10

  choice_metadata = candidate.metadata
  assert choice_metadata['safetyAttributes']['blocked'] is False


def test_chat_model_predict_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    model.predict(PROMPT, INVALID_PARAMETERS)


@pytest.mark.asyncio
async def test_chat_model_apredict(model):
  response = await model.apredict(PROMPT, VALID_PARAMETERS)

  assert len(response.responses) == 1
  assert len(response.responses[0].candidates) == 1

  candidate = response.responses[0].candidates[0]

  assert len(candidate.message.content) > 10

  choice_metadata = candidate.metadata
  assert choice_metadata['safetyAttributes']['blocked'] is False


@pytest.mark.asyncio
async def test_chat_model_apredict_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    await model.apredict(PROMPT, INVALID_PARAMETERS)


def test_chat_model_predict_stream(model):
  responses = list(model.predict_stream(PROMPT, VALID_PARAMETERS))

  assert len(responses) > 1
  for response in responses:
    assert len(response.responses) == 1
    assert len(response.responses[0].candidates) == 1

    candidate = response.responses[0].candidates[0]
    assert len(candidate.message.content) >= 1


def test_chat_model_predict_stream_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    list(model.predict_stream(PROMPT, INVALID_PARAMETERS))


@pytest.mark.asyncio
async def test_chat_model_apredict_stream(model):
  responses = [
      res async for res in model.apredict_stream(PROMPT, VALID_PARAMETERS)
  ]

  assert len(responses) > 1
  for response in responses:
    assert len(response.responses) == 1
    assert len(response.responses[0].candidates) == 1

    candidate = response.responses[0].candidates[0]
    assert len(candidate.message.content) >= 1


@pytest.mark.asyncio
async def test_chat_model_apredict_stream_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    async for _ in model.apredict_stream(PROMPT, INVALID_PARAMETERS):
      pass
