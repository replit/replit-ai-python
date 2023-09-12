from dataclasses import dataclass
from replit.ai.modelfarm.completion_model import (
    CompletionModel,
    CompletionModelResponse,
)
from typing import List, Dict, Any, Iterator
from replit.ai.modelfarm.google.utils import ready_parameters
from replit.ai.modelfarm.google.structs import TextGenerationResponse


class TextGenerationModel:
    """
    Class representing a Google completion model.

    Methods:
        from_pretrained - Loads a pretrained model using its identifier
        predict - completes a human-like text given an initial prompt.
        async_predict - Async version of the predict method.
    """

    def __init__(self, model_id: str):
        """Constructor method to initialize a text generation model."""
        self.underlying_model = CompletionModel(model_id)

    @staticmethod
    def from_pretrained(model_id: str) -> "TextGenerationModel":
        """
        Creates a Tokenizer from a pretrained model.

        Args:
            model_id (str): The identifier of the pretrained model.

        Returns:
            The TextGenerationModel class instance.
        """
        return TextGenerationModel(model_id)

    def predict(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        completes a human-like text given an initial prompt.

        Args:
            prompt (str): The initial text to start the generation.

        Returns:
            TextGenerationResponse: The model's response containing the completed text.
        """
        parameters = ready_parameters(kwargs)
        response = self.underlying_model.complete([prompt], **parameters)
        return self.__ready_response(response)

    def predict_streaming(
        self, prompt: str, **kwargs
    ) -> Iterator[TextGenerationResponse]:
        """
        completes a human-like text given an initial prompt.

        Args:
            prompt (str): The initial text to start the generation.

        Returns:
            TextGenerationResponse: The model's response containing the completed text.
        """
        parameters = ready_parameters(kwargs)
        response = self.underlying_model.stream_complete([prompt], **parameters)
        for x in response:
            yield self.__ready_response(x)

    async def async_predict(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Async version of the predict method. Equivalent to the predict method, but suited for asynchronous programming.

        Args:
            prompt (str): The initial text to start the generation.

        Returns:
            TextGenerationResponse: The model's response containing the completed text.
        """
        parameters = ready_parameters(kwargs)
        response = await self.underlying_model.async_complete([prompt], **parameters)
        return self.__ready_response(response)

    async def async_predict_streaming(
        self, prompt: str, **kwargs
    ) -> TextGenerationResponse:
        """
        Async version of the predict method. Equivalent to the predict method, but suited for asynchronous programming.

        Args:
            prompt (str): The initial text to start the generation.

        Returns:
            TextGenerationResponse: The model's response containing the completed text.
        """
        parameters = ready_parameters(kwargs)
        response = self.underlying_model.async_stream_complete([prompt], **parameters)
        async for x in response:
            yield self.__ready_response(x)

    def __ready_response(
        self, response: CompletionModelResponse
    ) -> TextGenerationResponse:
        """
        Transforms Completion Model's response into a readily usable format.

        Args:
            response (CompletionModelResponse): The original response from the underlying model.

        Returns:
            TextGenerationResponse: The transformed response.
        """
        choice = response.responses[0].choices[0]
        safetyAttributes = choice.metadata["safetyAttributes"]
        safetyCategories = dict(
            zip(safetyAttributes["categories"], safetyAttributes["scores"], strict=True)
        )

        return TextGenerationResponse(
            is_blocked=safetyAttributes["blocked"],
            raw_prediction_response=choice.model_dump(),
            safety_attributes=safetyCategories,
            text=choice.content,
        )
