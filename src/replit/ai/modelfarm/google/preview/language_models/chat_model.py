from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from replit.ai.modelfarm import AsyncModelfarm, Modelfarm
from replit.ai.modelfarm.google.structs import TextGenerationResponse
from replit.ai.modelfarm.google.utils import ready_parameters
from replit.ai.modelfarm.structs.chat import (
    ChatCompletionMessageRequestParam,
    ChatCompletionResponse,
    ChatCompletionStreamChunkResponse,
)

USER_AUTHOR = "user"
MODEL_AUTHOR = "bot"


@dataclass
class InputOutputTextPair:
    input_text: str
    output_text: str


@dataclass
class ChatMessage:
    content: str
    author: str


class ChatSession:
    context: Optional[str]
    examples: List[InputOutputTextPair]
    message_history: List[ChatMessage]
    underlying_model: str
    parameters: Dict[str, Any]

    _client: Modelfarm
    _async_client: AsyncModelfarm

    def __init__(
        self,
        underlying_model,
        context=None,
        examples: Optional[List[InputOutputTextPair]] = None,
        message_history: Optional[List[ChatMessage]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.examples = examples or []
        self.message_history = message_history or []
        self.underlying_model = underlying_model
        self.parameters = parameters or {}

        self._client = Modelfarm()
        self._async_client = AsyncModelfarm()

    def send_message(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(
            **self.parameters, **kwargs, **{
                "context": self.context,
                "examples": self.__build_chat_examples_from_io(),
            })
        response = self._client.chat.completions.create(
            model=self.underlying_model,
            messages=self.__build_replit_messages_from_history(),
            stream=False,
            **ready_parameters(predictParams),
        )
        self.add_model_message(self.__get_response_content(response))
        return self.__ready_response(response)

    async def async_send_message(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(
            **self.parameters, **kwargs, **{
                "context": self.context,
                "examples": self.__build_chat_examples_from_io(),
            })
        response = await self._async_client.chat.completions.create(
            model=self.underlying_model,
            messages=self.__build_replit_messages_from_history(),
            stream=False,
            **ready_parameters(predictParams),
        )
        self.add_model_message(self.__get_response_content(response))
        return self.__ready_response(response)

    def send_message_stream(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(
            **self.parameters, **kwargs, **{
                "context": self.context,
                "examples": self.__build_chat_examples_from_io(),
            })
        response = self._client.chat.completions.create(
            model=self.underlying_model,
            messages=self.__build_replit_messages_from_history(),
            stream=True,
            **ready_parameters(predictParams),
        )
        message = ""
        for chunk in response:
            transformedResponse = self.__ready_response(chunk)
            message += transformedResponse.text
            yield transformedResponse
        self.add_model_message(message)

    async def async_send_message_stream(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(
            **self.parameters, **kwargs, **{
                "context": self.context,
                "examples": self.__build_chat_examples_from_io(),
            })
        response = await self._async_client.chat.completions.create(
            model=self.underlying_model,
            messages=self.__build_replit_messages_from_history(),
            stream=True,
            **ready_parameters(predictParams),
        )
        message = ""
        async for chunk in response:
            transformedResponse = self.__ready_response(chunk)
            message += transformedResponse.text
            yield transformedResponse
        self.add_model_message(message)

    def add_user_message(self, message: str):
        chatMessage = ChatMessage(content=message, author=USER_AUTHOR)
        self.message_history.append(chatMessage)

    def add_model_message(self, message: str):
        chatMessage = ChatMessage(content=message, author=MODEL_AUTHOR)
        self.message_history.append(chatMessage)

    def __build_chat_examples_from_io(self) -> List[Dict[str, Dict]]:
        return [{
            "input": {
                "content": io.input_text,
                "author": ""
            },
            "output": {
                "content": io.output_text,
                "author": ""
            },
        } for io in self.examples]

    def __build_replit_messages_from_history(
            self) -> List[ChatCompletionMessageRequestParam]:
        return [
            self.__build_replit_message_from_google_chat_message(x)
            for x in self.message_history
        ]

    @staticmethod
    def __build_replit_message_from_google_chat_message(
        msg: ChatMessage, ) -> ChatCompletionMessageRequestParam:
        return {"content": msg.content, "role": msg.author}

    def __get_response_content(
        self, response: Union[ChatCompletionResponse,
                              ChatCompletionStreamChunkResponse]
    ) -> str:
        if isinstance(response, ChatCompletionResponse):
            return response.choices[0].message.content or ""
        return response.choices[0].delta.content or ""

    def __ready_response(
        self, response: Union[ChatCompletionResponse,
                              ChatCompletionStreamChunkResponse]
    ) -> TextGenerationResponse:
        """
        Transforms Completion Model's response into a readily usable format.

        Args:
            response (CompletionModelResponse): The original response from
                the underlying model.

        Returns:
            TextGenerationResponse: The transformed response.
        """
        choice = response.choices[0]
        text = self.__get_response_content(response)
        safetyAttributes = choice.metadata[
            "safetyAttributes"] if choice.metadata else {}
        safetyCategories = dict(
            zip(safetyAttributes["categories"],
                safetyAttributes["scores"],
                strict=True)) if safetyAttributes else {}
        return TextGenerationResponse(
            is_blocked=safetyAttributes["blocked"],
            raw_prediction_response=choice.model_dump(),
            safety_attributes=safetyCategories,
            text=text,
        )


class ChatModel:

    def __init__(self, model_id: str):
        self.underlying_model = model_id

    @staticmethod
    def from_pretrained(model_id: str) -> "ChatModel":
        return ChatModel(model_id)

    def start_chat(
        self,
        context: Optional[str] = "",
        examples: Optional[List[InputOutputTextPair]] = None,
        message_history: Optional[List[ChatMessage]] = None,
    ) -> ChatSession:
        chat_session = ChatSession(self.underlying_model, context, examples
                                   or [], message_history or [])
        return chat_session


#   def get_embeddings(self, content: List[str]) -> List[TextEmbedding]:
#     request = self.__ready_input(content)
#     response = self.underlying_model.embed(request, {})
#     return self.__ready_response(response)

#   async def async_get_embeddings(self,
#                                  content: List[str]) -> List[TextEmbedding]:
#     request = self.__ready_input(content)
#     response = await self.underlying_model.aembed(request, {})
#     return self.__ready_response(response)

#   def __ready_input(self, content: List[str]) -> List[Dict[str, Any]]:
#     return [{'content': x} for x in content]

#   def __ready_response(
#       self, response: EmbeddingModelResponse) -> List[TextEmbedding]:

#     def transform_response(x):
#       metadata = x.tokenCountMetadata
#       tokenCount = metadata.unbilledTokens + metadata.billableTokens
#       stats = TextEmbeddingStatistics(tokenCount, x.truncated)
#       return TextEmbedding(stats, x.values)

#     return [transform_response(x) for x in response.embeddings]
