from typing import Any, Dict, List, Optional, TypeAlias, Union

from pydantic import BaseModel

from .google import GoogleEmbeddingMetadata
from .shared import Usage

##
# Request params
##

InputParameter: TypeAlias = Union[str, List[str], List[int], List[List[int]]]

##
# Response models
##


class Embedding(BaseModel):
    object: str
    embedding: List[float]
    index: int
    metadata: Optional[Dict[str, Any]]


class EmbeddingModelResponse(BaseModel):
    object: str
    data: List[Embedding]
    model: str
    usage: Optional[Usage]
    metadata: Optional[GoogleEmbeddingMetadata]
