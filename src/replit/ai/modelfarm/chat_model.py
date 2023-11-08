from typing import Any, Dict, Iterator, List, Optional

import aiohttp
import requests

from .model import Model
from .structs import ChatMessage, ChatModelResponse


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

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> ChatModelResponse:
        """
        Makes a generation based on the messages and parameters.

        Args:
          messages (List[ChatMessage]): The list of messages in the conversation 
            so far.
          max_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          ChatModelResponse: The response from the model.

        """
        response = requests.post(
            self.server_url + "/v1beta2/chat",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ),
        )

        self._check_response(response)
        return ChatModelResponse(**response.json())

    async def async_chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> ChatModelResponse:
        """
        Makes an asynchronous generation based on the messages and parameters.

        Args:
          messages (List[ChatMessage]): The list of messages in the conversation 
            so far.
          max_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          ChatModelResponse: The response from the model.
        """
        async with aiohttp.ClientSession() as session, session.post(
            self.server_url + "/v1beta2/chat",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ),
        ) as response:
            await self._check_aresponse(response)
            return ChatModelResponse(**await response.json())

    def stream_chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> Iterator[ChatModelResponse]:
        """
        Streams generations based on the prompts and parameters.

        Args:
          messages (List[ChatMessage]): The list of messages in the conversation 
            so far.
          max_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          Iterator[ChatModelResponse]: An iterator of the responses from the model.
        """
        response = requests.post(
            self.server_url + "/v1beta2/chat_streaming",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ),
            stream=True,
        )
        self._check_streaming_response(response)
        for chunk in self._parse_streaming_response(response):
            yield ChatModelResponse(**chunk)

    async def async_stream_chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> Iterator[ChatModelResponse]:
        """
        Streams asynchronous generations based on the prompts and parameters.

        Args:
          messages (List[ChatMessage]): The list of messages in the conversation 
            so far.
          max_output_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
          temperature (float): The temperature of the generation. Defaults to 0.2.

        Returns:
          Iterator[ChatModelResponse]: An iterator of the responses from the model.
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session, session.post(
            self.server_url + "/v1beta2/chat_streaming",
            headers=self._get_auth_headers(),
            json=self.__build_request_payload(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ),
        ) as response:
            await self._check_streaming_aresponse(response)

            async for chunk in self._parse_streaming_aresponse(response):
                yield ChatModelResponse(**chunk)

    def __build_request_payload(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int],
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Builds the request payload.

        Args:
          messages (List[ChatMessage]): The list of messages in the conversation 
            so far.
          max_tokens (int): The maximum number of tokens to generate.
          temperature (float): The temperature of the generation.

        Returns:
          Dict[str, Any]: The request payload.
        """

        return {
            "model": self.model_name,
            "parameters": {
                "messages": [x.model_dump() for x in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        }
