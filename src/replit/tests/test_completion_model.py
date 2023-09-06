import pytest
from replit.ai import CompletionModel
from replit.ai.exceptions import BadRequestException

# module level constants
PROMPT = ["1 + 1 = "]
LONG_PROMPT = [
    "A very long answer to the question of what is the meaning of life is "
]
VALID_PARAMETERS = {"temperature": 0.5}
INVALID_PARAMETERS = {"invalid_parameter": 0.5}


# fixture for creating CompletionModel
@pytest.fixture
def model():
  return CompletionModel("text-bison")


def test_completion_model_generate(model):
  response = model.generate(PROMPT, VALID_PARAMETERS)

  assert len(response.responses) == 1
  assert len(response.responses[0].choices) == 1

  choice = response.responses[0].choices[0]

  assert '2' in choice.content

  choice_metadata = choice.metadata
  assert choice_metadata['safetyAttributes']['blocked'] is False


def test_completion_model_generate_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    model.generate(PROMPT, INVALID_PARAMETERS)


@pytest.mark.asyncio
async def test_completion_model_async_generate(model):
  response = await model.async_generate(PROMPT, VALID_PARAMETERS)

  assert len(response.responses) == 1
  assert len(response.responses[0].choices) == 1

  choice = response.responses[0].choices[0]

  assert '2' in choice.content

  choice_metadata = choice.metadata
  assert choice_metadata['safetyAttributes']['blocked'] is False


@pytest.mark.asyncio
async def test_completion_model_async_generate_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    await model.async_generate(PROMPT, INVALID_PARAMETERS)


def test_completion_model_generate_stream(model):
  responses = list(model.generate_stream(LONG_PROMPT, VALID_PARAMETERS))

  assert len(responses) > 1
  for response in responses:
    assert len(response.responses) == 1
    assert len(response.responses[0].choices) == 1

    choice = response.responses[0].choices[0]

    assert choice.content is not None


def test_completion_model_generate_stream_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    list(model.generate_stream(PROMPT, INVALID_PARAMETERS))


@pytest.mark.asyncio
async def test_completion_model_async_generate_stream(model):
  responses = [
      res async for res in model.async_generate_stream(LONG_PROMPT, VALID_PARAMETERS)
  ]

  assert len(responses) > 1
  for response in responses:
    assert len(response.responses) == 1
    assert len(response.responses[0].choices) == 1

    choice = response.responses[0].choices[0]

    assert choice.content is not None


@pytest.mark.asyncio
async def test_completion_model_async_generate_stream_invalid_parameter(model):
  with pytest.raises(BadRequestException):
    async for _ in model.async_generate_stream(LONG_PROMPT, INVALID_PARAMETERS):
      pass
