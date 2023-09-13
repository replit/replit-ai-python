from .model import Model
from typing import List, Dict, Any, Iterator
import requests
from .structs import ChatModelResponse, ChatSession, EmbeddingModelResponse
import aiohttp


class EmbeddingModel(Model):
  """Handles predictions from a embedding model."""

  def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
    """
        Initializes a EmbeddingModel instance.

        Args:
          model_name (str): The name of the model.
          **kwargs (Dict[str, Any]): Additional keyword arguments.
        """
    super().__init__(**kwargs)
    self.model_name = model_name

  def embed(self, content: List[Dict[str, Any]],
            **kwargs) -> EmbeddingModelResponse:
    """
        Makes a prediction based on the content and parameters.

        Args:
          content (List[Dict[str, Any]]): The list of content to embed.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          EmbeddingModelResponse: The response from the model.
        """
    response = requests.post(
        self.server_url + "/v1beta/embedding",
        headers=self._get_auth_headers(),
        json=self.__build_request_payload(content, **kwargs),
    )
    self._check_response(response)
    return EmbeddingModelResponse(**response.json())

  async def async_embed(self, content: List[Dict[str, Any]],
                        **kwargs: Dict[str, Any]) -> EmbeddingModelResponse:
    """
        Makes an asynchronous embedding generation based on the content and parameters.

        Args:
          content (List[Dict[str, Any]]): The list of content to embed. For most models the dictionary should contain the content to embed with the "content" key.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          ChatModelResponse: The response from the model.
        """
    async with aiohttp.ClientSession() as session:
      async with session.post(
          self.server_url + "/v1beta/embedding",
          headers=self._get_auth_headers(),
          json=self.__build_request_payload(content, **kwargs),
      ) as response:
        await self._check_aresponse(response)
        return EmbeddingModelResponse(**await response.json())

  def __build_request_payload(self, content: List[Dict[str, Any]],
                              **kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
        Builds the request payload.

        Args:
          content (List[Dict[str, Any]]): The list of content to embed. For most models the dictionary should contain the content to embed with the "content" key.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          Dict[str, Any]: The request payload.
        """

    return {
        "model": self.model_name,
        "parameters": {
            "content": content,
            **kwargs
        }
    }
