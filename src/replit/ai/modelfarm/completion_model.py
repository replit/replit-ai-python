from .model import Model
from typing import List, Dict, Any, Iterator
import requests
from .structs import CompletionModelResponse
import aiohttp


class CompletionModel(Model):
    """Handles predictions from a completion model."""

    def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
        """
        Initializes a CompletionModel instance.

        Args:
          model_name (str): The name of the model.
          **kwargs (Dict[str, Any]): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_name = model_name

    def complete(
        self,
        prompts: List[str],
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Dict[str, Any]
    ) -> CompletionModelResponse:
        """
        Makes a generation based on the prompts and parameters.

        Args:
          prompts (List[str]): The list of prompts.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          CompletionModelResponse: The response from the model.
        """
        response = requests.post(
            self.server_url + "/v1beta/completion",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                prompts,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                **kwargs
            ),
        )
        self._check_response(response)
        return CompletionModelResponse(**response.json())

    async def async_complete(
        self,
        prompts: List[str],
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> CompletionModelResponse:
        """
        Makes an asynchronous generation based on the prompts and parameters.

        Args:
          prompts (List[str]): The list of prompts.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          CompletionModelResponse: The response from the model.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.server_url + "/v1beta/completion",
                headers=self._get_auth_headers(),
                json=self.__build_request_payload(
                    prompts,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    **kwargs
                ),
            ) as response:
                await self._check_aresponse(response)
                return CompletionModelResponse(**await response.json())

    def stream_complete(
        self,
        prompts: List[str],
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Dict[str, Any]
    ) -> Iterator[CompletionModelResponse]:
        """
        Streams generations based on the prompts and parameters.

        Args:
          prompts (List[str]): The list of prompts.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          Iterator[CompletionModelResponse]: An iterator of the responses from the model.
        """
        response = requests.post(
            self.server_url + "/v1beta/completion_streaming",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                prompts,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                **kwargs
            ),
            stream=True,
        )
        self._check_streaming_response(response)
        for chunk in self._parse_streaming_response(response):
            yield CompletionModelResponse(**chunk)

    async def async_stream_complete(
        self,
        prompts: List[str],
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Dict[str, Any]
    ) -> Iterator[CompletionModelResponse]:
        """
        Streams asynchronous predictions based on the prompts and parameters.

        Args:
          prompts (List[str]): The list of prompts.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          Iterator[CompletionModelResponse]: An iterator of the responses from the model.
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            async with session.post(
                self.server_url + "/v1beta/completion_streaming",
                headers=self._get_auth_headers(),
                json=self.__build_request_payload(
                    prompts,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    **kwargs
                ),
            ) as response:
                await self._check_streaming_aresponse(response)

                async for chunk in self._parse_streaming_aresponse(response):
                    yield CompletionModelResponse(**chunk)

    def __build_request_payload(
        self, prompts: List[str], max_output_tokens: int, temperature: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Builds the request payload.

        Args:
          prompts (List[str]): The list of prompts.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          Dict[str, Any]: The request payload.
        """
        return {
            "model": self.model_name,
            "parameters": {
                "prompts": prompts,
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                **kwargs,
            },
        }
