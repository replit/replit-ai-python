from typing import Any, Dict, List, Optional, TypeAlias, Union

from pydantic import BaseModel

from .google import GoogleMetadata
from .shared import Usage

##
# Request params
##

PromptParameter: TypeAlias = Optional[Union[str, List[str], List[int],
                                            List[List[int]]]]

##
# Response models
##


class Choice(BaseModel):
    index: int
    text: str
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class CompletionModelResponse(BaseModel):
    id: str
    choices: List[Choice]
    model: str
    created: Optional[int] = None
    object: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: Optional[GoogleMetadata] = None
