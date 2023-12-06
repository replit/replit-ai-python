from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Literal,
    Optional,
    Union,
    overload,
)

from replit.ai.modelfarm.structs.completions import (
    CompletionModelResponse,
    PromptParameter,
)

if TYPE_CHECKING:
    from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


class Completions:
    _client: "Modelfarm"

    def __init__(self, client: "Modelfarm") -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: Literal[True],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[CompletionModelResponse]:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: Literal[False] = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionModelResponse:
        ...

    def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: bool = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[CompletionModelResponse, Iterator[CompletionModelResponse]]:
        """
        Makes a generation based on the messages and parameters.

        Args:
            model (str): The name of the model to use.
            prompt (PrompParameter): The prompt(s) to generate completion for.
            stream (bool): Whether to stream the responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float): The temperature of the generation. Defaults to 0.2.
            provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
                of the speficic provider. Defaults to None.

        Returns:
          If stream is True, returns an iterator of CompletionModelResponse.
          Otherwise, returns a CompletionModelResponse.

        """
        if stream:
            return self.__completion_stream(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            )
        return self.__completion(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_extra_parameters=provider_extra_parameters,
            **kwargs,
        )

    def __completion(
        self,
        model: str,
        prompt: PromptParameter,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionModelResponse:
        """
        Makes a generation based on prompt(s) and parameters.
        """
        response = self._client._post(
            "/v1beta2/completions",
            payload=_build_request_payload(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            ),
        )
        self._client._check_response(response)
        return CompletionModelResponse(**response.json())

    def __completion_stream(
        self,
        model: str,
        prompt: PromptParameter,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[CompletionModelResponse]:
        """
        Create a stream of CompletionModelResponse
        """
        response = self._client._post(
            "/v1beta2/completions",
            payload=_build_request_payload(
                model=model,
                prompt=prompt,
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
            yield CompletionModelResponse(**chunk)


class AsyncCompletions:
    _client: "AsyncModelfarm"

    def __init__(self, client: "AsyncModelfarm") -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: Literal[True],
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionModelResponse]:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: Literal[False] = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionModelResponse:
        ...

    async def create(
        self,
        *,
        model: str,
        prompt: PromptParameter,
        stream: bool = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.2,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[CompletionModelResponse,
               AsyncIterator[CompletionModelResponse]]:
        """
        Makes a generation based on the messages and parameters.

        Args:
            model (str): The name of the model to use.
            prompt (PromptParameter): The prompt(s) to generate completion for.
            stream (bool): Whether to stream the responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float): The temperature of the generation. Defaults to 0.2.
            provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
                of the speficic provider. Defaults to None.

        Returns:
          If stream is True, returns an iterator of CompletionModelResponse.
          Otherwise, returns a CompletionModelResponse.

        """
        if stream:
            return self.__completion_stream(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_extra_parameters=provider_extra_parameters,
                **kwargs,
            )
        return await self.__completion(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_extra_parameters=provider_extra_parameters,
            **kwargs,
        )

    async def __completion(
        self,
        model: str,
        prompt: PromptParameter,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionModelResponse:
        """
        Makes a generation based on the prompt(s) and parameters.
        """
        async with self._client._post(
                "/v1beta2/completions",
                payload=_build_request_payload(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    provider_extra_parameters=provider_extra_parameters,
                    **kwargs,
                ),
        ) as response:
            await self._client._check_response(response)
            return CompletionModelResponse(**await response.json())

    async def __completion_stream(
        self,
        model: str,
        prompt: PromptParameter,
        max_tokens: Optional[int],
        temperature: float,
        provider_extra_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionModelResponse]:
        """
        Create a stream of CompletionModelResponse
        """
        async with self._client._post(
                "/v1beta2/completions",
                payload=_build_request_payload(
                    model=model,
                    prompt=prompt,
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
                yield CompletionModelResponse(**chunk)


def _build_request_payload(
    model: str,
    prompt: PromptParameter,
    max_tokens: Optional[int],
    temperature: float,
    stream: bool,
    provider_extra_parameters: Optional[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Builds the request payload.

    Args:
        model (str): The name of the model to use.
        prompt (PromptParameter): The prompt(s) to generate completion for.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature of the generation.
        provider_extra_parameters (Optional[Dict[str, Any]]): Extra parameters
            of the speficic provider.

    Returns:
      Dict[str, Any]: The request payload.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "provider_extra_parameters": provider_extra_parameters,
        **kwargs,
    }

    # Drop any keys with a value of None
    payload = {k: v for k, v in payload.items() if v is not None}
    return payload
