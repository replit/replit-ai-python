from dataclasses import dataclass
from replit.ai.modelfarm.chat_model import (
    ChatModel as ReplitChatModel,
    ChatModelResponse,
)
from replit.ai.modelfarm.structs import (
    ChatSession as ReplitChatSession,
    ChatExample as ReplitChatExample,
    ChatMessage as ReplitChatMessage,
)
from typing import List, Optional, Dict, Any
from replit.ai.modelfarm.google.utils import ready_parameters
from replit.ai.modelfarm.google.structs import TextGenerationResponse

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
    underlying_model: ReplitChatModel
    parameters: Dict[str, Any]

    def __init__(
        self,
        underlying_model,
        context=None,
        examples: Optional[List[InputOutputTextPair]] = None,
        message_history: Optional[List[ChatMessage]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.context = context
        self.examples = examples = []
        self.message_history = message_history or []
        self.underlying_model = underlying_model
        self.parameters = parameters or {}

    def send_message(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(**self.parameters, **kwargs)
        session = self.get_chat_session()
        response = self.underlying_model.chat(
            [session], **ready_parameters(predictParams)
        )
        self.add_model_message(self.__get_response_content(response))
        return self.__ready_response(response)

    async def async_send_message(self, message: str, **kwargs):
        self.add_user_message(message)
        predictParams = dict(**self.parameters, **kwargs)
        session = self.get_chat_session()
        response = await self.underlying_model.async_chat(
            [session], **ready_parameters(predictParams)
        )
        self.add_model_message(self.__get_response_content(response))
        return self.__ready_response(response)

    def send_message_stream(self, message: str, **kwargs):
        newMessage = self.add_user_message(message)
        predictParams = dict(**self.parameters, **kwargs)
        session = self.get_chat_session()

        response = self.underlying_model.stream_chat(
            [session], **ready_parameters(predictParams)
        )
        message = ""
        for chunk in response:
            transformedResponse = self.__ready_response(chunk)
            message += transformedResponse.text
            yield transformedResponse
        self.add_model_message(message)

    async def async_send_message_stream(self, message: str, **kwargs):
        newMessage = self.add_user_message(message)
        predictParams = dict(**self.parameters, **kwargs)
        session = self.get_chat_session()
        response = self.underlying_model.async_stream_chat(
            [session], **ready_parameters(predictParams)
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

    def get_chat_session(self):
        examples = [self.__build_replit_chat_example_from_io(x) for x in self.examples]
        msgs = [
            self.__build_replit_message_from_google_chat_message(x)
            for x in self.message_history
        ]

        return ReplitChatSession(context=self.context, examples=examples, messages=msgs)

    @staticmethod
    def __build_replit_chat_example_from_io(io: InputOutputTextPair):
        return ReplitChatExample(
            input=ReplitChatMessage(content=io.input_text),
            output=ReplitChatMessage(content=io.output_text),
        )

    @staticmethod
    def __build_replit_message_from_google_chat_message(msg: ChatMessage):
        return ReplitChatMessage(content=msg.content, author=msg.author)

    def __get_response_content(self, response: ChatModelResponse) -> str:
        candidate = response.responses[0].candidates[0]
        return candidate.message.content

    def __ready_response(self, response: ChatModelResponse) -> TextGenerationResponse:
        """
        Transforms Completion Model's response into a readily usable format.

        Args:
            response (CompletionModelResponse): The original response from the underlying model.

        Returns:
            TextGenerationResponse: The transformed response.
        """
        candidate = response.responses[0].candidates[0]
        safetyAttributes = candidate.metadata["safetyAttributes"]
        safetyCategories = dict(
            zip(safetyAttributes["categories"], safetyAttributes["scores"], strict=True)
        )

        return TextGenerationResponse(
            is_blocked=safetyAttributes["blocked"],
            raw_prediction_response=candidate.model_dump(),
            safety_attributes=safetyCategories,
            text=candidate.message.content,
        )


class ChatModel:
    def __init__(self, model_id: str):
        self.underlying_model = ReplitChatModel(model_id)

    @staticmethod
    def from_pretrained(model_id: str) -> "ChatModel":
        return ChatModel(model_id)

    def start_chat(
        self,
        context: Optional[str] = "",
        examples: Optional[List[InputOutputTextPair]] = None,
        message_history: Optional[List[ChatMessage]] = None,
    ) -> ChatSession:
        chat_session = ChatSession(
            self.underlying_model, context, examples or [], message_history or []
        )
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
