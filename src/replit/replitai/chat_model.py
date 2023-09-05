from .model import Model
from typing import List, Dict, Any, Iterator
import requests
from .structs import ChatModelResponse, ChatSession
import aiohttp


class ChatModel(Model):
  """Handles predictions from a chat model."""

  def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
    """
    Initializes a ChatModel instance.

    Args:
      model_name (str): The name of the model.
      **kwargs (Dict[str, Any]): Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self.model_name = model_name

  def predict(self, prompts: List[ChatSession],
              parameters: Dict[str, Any]) -> ChatModelResponse:
    """
    Makes a prediction based on the prompts and parameters.

    Args:
      prompts (List[ChatSession]): The list of chat session prompts.
      parameters (Dict[str, Any]): The dictionary of parameters.

    Returns:
      ChatModelResponse: The response from the model.
    """
    response = requests.post(
        self.server_url + "/chat",
        headers=self._get_auth_headers(),
        json=self.__build_request_payload(prompts, parameters),
    )
    self._check_response(response)
    return ChatModelResponse(**response.json())

  async def apredict(self, prompts: List[str],
                     parameters: Dict[str, Any]) -> ChatModelResponse:
    """
    Makes an asynchronous prediction based on the prompts and parameters.

    Args:
      prompts (List[ChatSession]): The list of prompts.
      parameters (Dict[str, Any]): The dictionary of parameters.

    Returns:
      ChatModelResponse: The response from the model.
    """
    async with aiohttp.ClientSession() as session:
      async with session.post(
          self.server_url + "/chat",
          headers=self._get_auth_headers(),
          json=self.__build_request_payload(prompts, parameters),
      ) as response:
        await self._check_aresponse(response)
        return ChatModelResponse(**await response.json())

  def predict_stream(
      self, prompts: List[str],
      parameters: Dict[str, Any]) -> Iterator[ChatModelResponse]:
    """
    Streams predictions based on the prompts and parameters.

    Args:
      prompts (List[ChatSession]): The list of prompts.
      parameters (Dict[str, Any]): The dictionary of parameters.

    Returns:
      Iterator[ChatModelResponse]: An iterator of the responses from the model.
    """
    response = requests.post(
        self.server_url + "/chat_streaming",
        headers=self._get_auth_headers(),
        json=self.__build_request_payload(prompts, parameters),
        stream=True,
    )
    self._check_streaming_response(response)
    for chunk in self._parse_streaming_response(response):
      yield ChatModelResponse(**chunk)

  async def apredict_stream(
      self, prompts: List[str],
      parameters: Dict[str, Any]) -> Iterator[ChatModelResponse]:
    """
    Streams asynchronous predictions based on the prompts and parameters.

    Args:
      prompts (List[ChatSession]): The list of prompts.
      parameters (Dict[str, Any]): The dictionary of parameters.

    Returns:
      Iterator[ChatModelResponse]: An iterator of the responses from the model.
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
        total=15)) as session:
      async with session.post(
          self.server_url + "/chat_streaming",
          headers=self._get_auth_headers(),
          json=self.__build_request_payload(prompts, parameters),
      ) as response:
        await self._check_streaming_aresponse(response)

        async for chunk in self._parse_streaming_aresponse(response):
          yield ChatModelResponse(**chunk)

  def __build_request_payload(self, prompts: List[ChatSession],
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the request payload.

    Args:
      prompts (List[ChatSession]): The list of prompts.
      parameters (Dict[str, Any]): The dictionary of parameters.

    Returns:
      Dict[str, Any]: The request payload.
    """

    return {
        "model": self.model_name,
        "parameters": {
            "prompts": [x.model_dump() for x in prompts],
            **parameters
        }
    }
