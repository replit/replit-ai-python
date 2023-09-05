from requests import JSONDecodeError
from .replit_identity_token_manager import replit_identity_token_manager
from replit.ai.config import get_config
from .exceptions import InvalidResponseException, BadRequestException
import aiohttp
import requests
from typing import Union, Iterator, List
import json


class Model:
  server_url: str

  def __init__(self, **kwargs):
    self.server_url = kwargs.get('server_url') or get_config().rootUrl

  def predict(self, prompt, parameters):
    raise NotImplementedError(
        "Subclasses of Model must implement predict(self, prompt, parameters)")

  def _check_response(self, response):
    try:
      rjson = response.json()
    except JSONDecodeError as e:
      raise InvalidResponseException(
          f"Invalid response: {response.text}") from e
    if response.status_code == 400:
      raise BadRequestException(rjson['detail'])
    if response.status_code != 200:
      if 'detail' in rjson:
        raise InvalidResponseException(rjson['detail'])
      raise InvalidResponseException(rjson)

  def _check_streaming_response(self, response):
    if response.status_code == 200:
      return
    self._check_response(response)

  async def _check_aresponse(self, response):
    try:
      rjson = await response.json()
    except JSONDecodeError as e:
      raise InvalidResponseException(
          f"Invalid response: {response.text}") from e

    if response.status == 400:
      raise BadRequestException(rjson['detail'])
    if response.status != 200:
      if 'detail' in rjson:
        raise InvalidResponseException(rjson['detail'])
      raise InvalidResponseException(rjson)

  async def _check_streaming_aresponse(self, response):
    if response.status == 200:
      return
    await self._check_aresponse(response)

  def _get_auth_headers(self):
    token = replit_identity_token_manager.get_token()
    return {"Authorization": f"Bearer {token}"}

  def _parse_streaming_response(self, response) -> Iterator[any]:
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

  async def _parse_streaming_aresponse(self, response) -> Iterator[any]:
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
