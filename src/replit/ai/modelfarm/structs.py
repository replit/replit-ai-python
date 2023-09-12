from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel


class TokenCountMetadata(BaseModel):
    billableTokens: int = 0
    unbilledTokens: int = 0
    billableCharacters: int = 0
    unbilledCharacters: int = 0


class Metadata(BaseModel):
    inputTokenCount: Optional[TokenCountMetadata] = None
    outputTokenCount: Optional[TokenCountMetadata] = None


class CompletionModelRequest(BaseModel):
    model: str
    parameters: Dict[str, Any]


class Choice(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    choices: List[Choice]


class CompletionModelResponse(BaseModel):
    metadata: Optional[Metadata] = None
    responses: List[PromptResponse]


class ChatMessage(BaseModel):
    content: str
    author: str = ""


class ChatExample(BaseModel):
    input: ChatMessage
    output: ChatMessage


class ChatSession(BaseModel):
    context: str
    examples: List[ChatExample]
    messages: List[ChatMessage]


class ChatModelRequest(BaseModel):
    model: str
    parameters: Dict[str, Any]


class Candidate(BaseModel):
    message: ChatMessage
    metadata: Optional[Dict[str, Any]] = None


class ChatPromptResponse(BaseModel):
    candidates: List[Candidate]


class ChatModelResponse(BaseModel):
    metadata: Optional[Metadata] = None
    responses: List[ChatPromptResponse]


class Embedding(BaseModel):
    values: List[float]
    tokenCountMetadata: Optional[TokenCountMetadata] = None
    truncated: bool


class EmbeddingMetadata(BaseModel):
    tokenCountMetadata: Optional[TokenCountMetadata] = None


class EmbeddingModelRequest(BaseModel):
    model: str
    parameters: Dict[str, Any]


class EmbeddingModelResponse(BaseModel):
    metadata: Optional[EmbeddingMetadata] = None
    embeddings: List[Embedding]
