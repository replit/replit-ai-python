from typing import TYPE_CHECKING, Any, Dict, Optional

from replit.ai.modelfarm.structs.embeddings import (
    EmbeddingModelResponse,
    InputParameter,
)

if TYPE_CHECKING:
    from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


class Embeddings:
    _client: "Modelfarm"

    def __init__(self, client: "Modelfarm") -> None:
        self._client = client

    def create(
        self,
        *,
        input: InputParameter,
        model: str,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """
        Makes a prediction based on the input and parameters.

        Args:
          input (InputParameter): The input(s) to embed.
          model (str): The name of the model.

        Returns:
          EmbeddingModelResponse: The response from the model.
        """
        response = self._client._post(
            "/v1beta2/embeddings",
            payload=_build_request_payload(
                input,
                model,
                provider_extra_parameters,
                **kwargs,
            ),
        )
        self._client._check_response(response)
        return EmbeddingModelResponse(**response.json())


class AsyncEmbeddings:
    _client: "AsyncModelfarm"

    def __init__(self, client: "AsyncModelfarm") -> None:
        self._client = client

    async def create(
        self,
        *,
        input: InputParameter,
        model: str,
        provider_extra_parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """
            Makes an asynchronous embedding generation based on the input
            and parameters.
        
            Args:
                input (EmbeddingInput): The input(s) to embed.
                model (str): The name of the model.
        
            Returns:
                EmbeddingModelResponse: The response from the model.
            """
        async with self._client._post(
                "/v1beta2/embeddings",
                payload=_build_request_payload(
                    input,
                    model,
                    provider_extra_parameters,
                    **kwargs,
                ),
        ) as response:
            await self._client._check_response(response)
            return EmbeddingModelResponse(**await response.json())


def _build_request_payload(
    input: InputParameter,
    model: str,
    provider_extra_parameters: Optional[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Builds the request payload.
    
    Args:
      input (InputParameter): The input(s) to embed.
      model (str): The name of the model to use.
    
    Returns:
      Dict[str, Any]: The request payload.
    """

    params = {
        "model": model,
        "input": input,
        "provider_extra_parameters": provider_extra_parameters,
        **kwargs,
    }

    return {k: v for k, v in params.items() if v is not None}
