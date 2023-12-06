import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterator, Optional

import aiohttp
import requests
from aiohttp import ClientResponse
from requests import JSONDecodeError, Response

from .chat_completions import AsyncChat, Chat
from .completions import AsyncCompletions, Completions
from .config import get_config
from .embeddings import AsyncEmbeddings, Embeddings
from .exceptions import BadRequestException, InvalidResponseException
from .replit_identity_token_manager import ReplitIdentityTokenManager


class BaseModelfarm:

    def __init__(self, base_url: Optional[str] = None) -> None:
        """
        Initializes a new instance of the BaseModelfarm class.
        """
        self.base_url = base_url or get_config().rootUrl
        self.auth = ReplitIdentityTokenManager()

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Gets authentication headers required for API requests.

        Returns:
            dict: A dictionary containing the Authorization header.
        """
        token = self.auth.get_token()
        return {"Authorization": f"Bearer {token}"}


class Modelfarm(BaseModelfarm):
    chat: Chat
    embeddings: Embeddings
    completions: Completions

    def __init__(
        self,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the Modelfarm class.
        """
        super().__init__(base_url)

        self.chat = Chat(self)
        self.embeddings = Embeddings(self)
        self.completions = Completions(self)

    def _post(
        self,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Response:
        return requests.post(
            url=self.base_url + path,
            headers=self._get_auth_headers(),
            json=payload,
            stream=stream,
            **kwargs,
        )

    def _check_response(self, response: Response) -> None:
        """
        Validates a response from the server.

        Parameters:
            response: The server response to check.

        Raises:
            InvalidResponseException: If the response is not valid JSON.
            BadRequestException: If the response contains a 400 status code.
        """
        try:
            rjson = response.json()
        except JSONDecodeError as e:
            raise InvalidResponseException(
                f"Invalid response: {response.text}") from e

        if response.status_code == 400:
            raise BadRequestException(rjson["detail"])
        if response.status_code != 200:
            if "detail" in rjson:
                raise InvalidResponseException(rjson["detail"])
            raise InvalidResponseException(rjson)

    def _check_streaming_response(self, response: Response) -> None:
        """
        Validates a streaming response from the server.

        Parameters:
            response: The server's streaming response to check.
        """
        if response.status_code == 200:
            return
        self._check_response(response)

    def _parse_streaming_response(self, response) -> Iterator[Any]:
        """
        Parses a streaming response from the server.

        Parameters:
            response: The server's streaming response to parse.

        Yields:
            JSON objects extracted from the streaming response.
        """
        buffer = b""
        decoder = json.JSONDecoder()
        for chunk in response.iter_content(chunk_size=128):
            buffer += chunk
            buffer_str = buffer.decode("utf-8")

            start_idx = 0
            # Iteratively parse JSON objects
            while start_idx < len(buffer_str):
                try:
                    # Load JSON object
                    result = decoder.raw_decode(buffer_str, start_idx)
                    json_obj, end_idx = result

                    yield json_obj
                    # Update start index for next iteration
                    start_idx = end_idx
                except json.JSONDecodeError:
                    break
            buffer = buffer[start_idx:]


class AsyncModelfarm(BaseModelfarm):
    chat: AsyncChat
    embeddings: AsyncEmbeddings
    completions: AsyncCompletions

    def __init__(
        self,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the AsyncModelfarm class.
        """
        super().__init__(base_url)

        self.chat = AsyncChat(self)
        self.embeddings = AsyncEmbeddings(self)
        self.completions = AsyncCompletions(self)

    @asynccontextmanager
    async def _post(
        self,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 15,
        **kwargs,
    ) -> AsyncGenerator[ClientResponse, None]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
                total=timeout)) as session, session.post(
                    url=self.base_url + path,
                    headers=self._get_auth_headers(),
                    json=payload,
                    **kwargs) as response:
            yield response

    async def _check_response(self, response: ClientResponse) -> None:
        """
        Validates an asynchronous response from the server.

        Parameters:
            response: The asynchronous server response to check.

        Raises:
            InvalidResponseException: If the response is not valid JSON.
            BadRequestException: If the response contains a 400 status code.
        """
        try:
            rjson = await response.json()
        except JSONDecodeError as e:
            raise InvalidResponseException(
                f"Invalid response: {response.text}") from e

        if response.status == 400:
            raise BadRequestException(rjson["detail"])
        if response.status != 200:
            if "detail" in rjson:
                raise InvalidResponseException(rjson["detail"])
            raise InvalidResponseException(rjson)

    async def _check_streaming_response(self,
                                        response: ClientResponse) -> None:
        """
        Validates an asynchronous streaming response from the server.

        Parameters:
            response: The server's asynchronous streaming response to check.
        """
        if response.status == 200:
            return
        await self._check_response(response)

    async def _parse_streaming_response(
            self, response: ClientResponse) -> AsyncIterator[Any]:
        """
        Asynchronously parses a streaming response from the server.

        Parameters:
            response: The server's asynchronous streaming response to parse.

        Yields:
            JSON objects extracted from the streaming response.
        """
        buffer = b""
        decoder = json.JSONDecoder()
        while True:
            chunk = await response.content.read(128)
            if not chunk:
                break
            buffer += chunk
            buffer_str = buffer.decode("utf-8")

            start_idx = 0
            # Iteratively parse JSON objects
            while start_idx < len(buffer_str):
                try:
                    # Load JSON object
                    result = decoder.raw_decode(buffer_str, start_idx)
                    json_obj, end_idx = result

                    yield json_obj

                    # Update start index for next iteration
                    start_idx = end_idx
                except json.JSONDecodeError:
                    break
            buffer = buffer[start_idx:]
