import json
from typing import Iterator, List, Union

import aiohttp
import requests
from replit.ai.modelfarm.config import get_config
from requests import JSONDecodeError

from .exceptions import BadRequestException, InvalidResponseException
from .replit_identity_token_manager import ReplitIdentityTokenManager


class Model:
    """
    Base class for models.

    Attributes:
        server_url (str): The URL of the server to which the model sends requests.

    Methods:
        predict(prompt, parameters): Abstract method to be implemented by subclasses.
    """

    server_url: str

    def __init__(self, **kwargs):
        """
        Initialize a Model object with an optional server URL.

        Keyword Arguments:
            server_url (str): Optional. Server URL for the model.
                              Defaults to the value in the configuration.
        """
        self.server_url = kwargs.get("server_url") or get_config().rootUrl
        self.auth = ReplitIdentityTokenManager()

    def _check_response(self, response):
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
            raise InvalidResponseException(f"Invalid response: {response.text}") from e

        if response.status_code == 400:
            raise BadRequestException(rjson["detail"])
        if response.status_code != 200:
            if "detail" in rjson:
                raise InvalidResponseException(rjson["detail"])
            raise InvalidResponseException(rjson)

    def _check_streaming_response(self, response):
        """
        Validates a streaming response from the server.

        Parameters:
            response: The server's streaming response to check.
        """
        if response.status_code == 200:
            return
        self._check_response(response)

    async def _check_aresponse(self, response):
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
            raise InvalidResponseException(f"Invalid response: {response.text}") from e

        if response.status == 400:
            raise BadRequestException(rjson["detail"])
        if response.status != 200:
            if "detail" in rjson:
                raise InvalidResponseException(rjson["detail"])
            raise InvalidResponseException(rjson)

    async def _check_streaming_aresponse(self, response):
        """
        Validates an asynchronous streaming response from the server.

        Parameters:
            response: The server's asynchronous streaming response to check.
        """
        if response.status == 200:
            return
        await self._check_aresponse(response)

    def _get_auth_headers(self):
        """
        Gets authentication headers required for API requests.

        Returns:
            dict: A dictionary containing the Authorization header.
        """
        token = self.auth.get_token()
        return {"Authorization": f"Bearer {token}"}

    def _parse_streaming_response(self, response) -> Iterator[any]:
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

    async def _parse_streaming_aresponse(self, response) -> Iterator[any]:
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
