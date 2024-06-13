from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    overload,
)

from replit.ai.modelfarm.structs.chat import (
    ChatCompletionMessageRequestParam,
    ChatCompletionResponse,
    ChatCompletionStreamChunkResponse,
)

if TYPE_CHECKING:
    from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


class Completions:
    _client: "Modelfarm"

    def __init__(self, client: "Modelfarm") -> None:
        """
        Initializes a new instance of the Completions class.
        """
        self._client = client

    @overload
    def create(
        self,
        *,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: Literal[True],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionStreamChunkResponse]:
        ...

    @overload
    def create(
        self,
        *,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: Literal[False] = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        ...

    def create(
        self,
        *,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: bool = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletionResponse,
               Iterator[ChatCompletionStreamChunkResponse]]:
        """
        Makes a generation based on the messages and parameters.

        Args:
            messages (List[ChatCompletionMessageRequestParam]): The list of messages 
                in the conversation so far.
            model (str): The name of the model to use.
            stream (bool): Whether to stream the responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float): The temperature of the generation. Defaults to 0.2.
            provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
                of the speficic provider. Defaults to None.

        Returns:
          If stream is True, returns an iterator of ChatCompletionStreamChunkResponse.
          Otherwise, returns a ChatCompletionResponse.

        """
        if stream:
            return self.__chat_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            )
        return self.__chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_extra_parameters=provider_extra_parameters,
            **kwargs,
        )

    def __chat(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        response = self._client._post(
            "/v1beta2/chat/completions",
            payload=_build_request_payload(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            ),
        )
        self._client._check_response(response)
        return ChatCompletionResponse(**response.json())

    def __chat_stream(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionStreamChunkResponse]:
        """
        Create a stream of ChatCompletionStreamChunkResponse
        """
        response = self._client._post(
            "/v1beta2/chat/completions",
            payload=_build_request_payload(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            ),
            stream=True,
        )
        self._client._check_streaming_response(response)
        for chunk in self._client._parse_streaming_response(response):
            yield ChatCompletionStreamChunkResponse(**chunk)


class AsyncCompletions:
    _client: "AsyncModelfarm"

    def __init__(self, client: "AsyncModelfarm") -> None:
        """
        Initializes a new instance of the Completions class.
        """
        self._client = client

    @overload
    async def create(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: Literal[True],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionStreamChunkResponse]:
        ...

    @overload
    async def create(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: Literal[False] = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        ...

    async def create(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        stream: bool = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletionResponse,
               AsyncIterator[ChatCompletionStreamChunkResponse]]:
        """
        Makes a generation based on the messages and parameters.

        Args:
            messages (List[ChatCompletionMessageRequestParam]): The list of messages
                in the conversation so far.
            model (str): The name of the model to use.
            stream (bool): Whether to stream the responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float): The temperature of the generation. Defaults to 0.2.
            provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
                of the speficic provider. Defaults to None.

        Returns:
          If stream is True, returns an iterator of ChatCompletionStreamChunkResponse.
          Otherwise, returns a ChatCompletionResponse.

        """
        if stream:
            return self.__chat_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            )
        return await self.__chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_extra_parameters=provider_extra_parameters,
            **kwargs,
        )

    async def __chat(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        async with self._client._post(
                "/v1beta2/chat/completions",
                payload=_build_request_payload(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    provider_extra_parameters=provider_extra_parameters,
                    **kwargs,
                ),
        ) as response:
            await self._client._check_response(response)
            return ChatCompletionResponse(**await response.json())

    async def __chat_stream(
        self,
        messages: List[ChatCompletionMessageRequestParam],
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionStreamChunkResponse]:
        """
        Create a stream of ChatCompletionStreamChunkResponse
        """
        async with self._client._post(
                "/v1beta2/chat/completions",
                payload=_build_request_payload(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    provider_extra_parameters=provider_extra_parameters,
                    **kwargs,
                ),
        ) as response:
            await self._client._check_streaming_response(response)
            async for chunk in self._client._parse_streaming_response(
                    response):
                yield ChatCompletionStreamChunkResponse(**chunk)


class Chat:
    completions: Completions

    def __init__(self, client: "Modelfarm") -> None:
        """
        Initializes a new instance of the Chat class.
        """
        self.completions = Completions(client)


class AsyncChat:
    completions: AsyncCompletions

    def __init__(self, client: "AsyncModelfarm") -> None:
        """
        Initializes a new instance of the AsyncChat class.
        """
        self.completions = AsyncCompletions(client)


def _build_request_payload(
    messages: List[ChatCompletionMessageRequestParam],
    model: str,
    max_tokens: Optional[int],
    temperature: float,
    stream: bool,
    provider_extra_parameters: Optional[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Builds the request payload.

    Args:
        messages (List[ChatCompletionMessageRequestParam]): The list of messages 
            in the conversation so far.
        model (str): The name of the model to use.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature of the generation.
        provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
            of the speficic provider.

    Returns:
      Dict[str, Any]: The request payload.
    """

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "provider_extra_parameters": provider_extra_parameters,
        **kwargs,
    }

    # Drop any keys with a value of None
    payload = {k: v for k, v in payload.items() if v is not None}
    return payload
