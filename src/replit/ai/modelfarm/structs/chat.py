from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import Required, TypedDict

from .google import GoogleMetadata
from .shared import Usage

##
# Request params
##


class ChatCompletionMessageRequestParam(TypedDict, total=False):
    role: Required[str]
    content: Optional[str]
    tool_calls: Optional[List]
    tool_call_id: Optional[str]


##
# Response models
##


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall


class ChoiceMessage(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class BaseChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Choice(BaseChoice):
    message: ChoiceMessage


class ChoiceStream(BaseChoice):
    delta: ChoiceMessage


class BaseChatCompletionResponse(BaseModel):
    id: str
    choices: List[BaseChoice]
    model: str
    created: Optional[int]
    object: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: Optional[GoogleMetadata] = None


class ChatCompletionResponse(BaseChatCompletionResponse):
    choices: List[Choice]
    object: Optional[str] = "chat.completion"


class ChatCompletionStreamChunkResponse(BaseChatCompletionResponse):
    choices: List[ChoiceStream]
    object: Optional[str] = "chat.completion.chunk"
