from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TokenCountMetadata(BaseModel):
    billableTokens: int = 0
    unbilledTokens: int = 0
    billableCharacters: int = 0
    unbilledCharacters: int = 0


class GoogleMetadata(BaseModel):
    inputTokenCount: Optional[TokenCountMetadata] = None
    outputTokenCount: Optional[TokenCountMetadata] = None


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CompletionModelRequest(BaseModel):
    model: str
    parameters: Dict[str, Any]


class Choice(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    choices: List[Choice]


class CompletionModelResponse(BaseModel):
    metadata: Optional[GoogleMetadata] = None
    responses: List[PromptResponse]


class ChatMessage(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None


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


class Choice(BaseModel):
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[ChatMessage] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatModelResponse(BaseModel):
    id: str
    choices: List[Choice]
    model: str
    created: Optional[int]
    object: Optional[str]
    usage: Optional[Usage]
    metadata: Optional[GoogleMetadata]


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
