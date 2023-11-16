from typing import Any, Dict, List, TypeAlias, Union

import aiohttp
import requests

from .model import Model
from .structs import EmbeddingModelResponse

EmbeddingInput: TypeAlias = Union[str, List[str]]


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

    def embed(self, input: EmbeddingInput, **kwargs) -> EmbeddingModelResponse:
        """
        Makes a prediction based on the input and parameters.

        Args:
          input (EmbeddingInput): The input(s) to embed.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          EmbeddingModelResponse: The response from the model.
        """
        response = requests.post(
            self.server_url + "/v1beta2/embedding",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(input, **kwargs),
        )
        self._check_response(response)
        return EmbeddingModelResponse(**response.json())

    async def async_embed(self, input: EmbeddingInput,
                          **kwargs: Dict[str, Any]) -> EmbeddingModelResponse:
        """
        Makes an asynchronous embedding generation based on the input and parameters.

        Args:
          input (EmbeddingInput): The input(s) to embed.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          EmbeddingModelResponse: The response from the model.
        """
        async with aiohttp.ClientSession() as session, session.post(
                self.server_url + "/v1beta2/embedding",
                headers=self._get_auth_headers(),
                json=self.__build_request_payload(input, **kwargs),
        ) as response:
            await self._check_aresponse(response)
            return EmbeddingModelResponse(**await response.json())

    def __build_request_payload(self, input: EmbeddingInput,
                                **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds the request payload.

        Args:
          input (EmbeddingInput): The input(s) to embed.
          parameters (Dict[str, Any]): The dictionary of parameters.

        Returns:
          Dict[str, Any]: The request payload.
        """

        return {
            "model": self.model_name,
            "parameters": {
                "input": input,
                **kwargs
            }
        }
