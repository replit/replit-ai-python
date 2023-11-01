from typing import Optional, List, Union, Dict, Any, Literal
from pydantic import BaseModel, Field


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


class ImageGenerationModelImageResult(BaseModel):
    url: str
    content_type: str
    file_name: str
    file_size: int
    width: int
    height: int


class ImageGenerationModelResponse(BaseModel):
    images: List[ImageGenerationModelImageResult]
    seed: int


class ImageSize(BaseModel):
    width: int = Field(
        default=512, description="The width of the generated image.", gt=0, le=4096
    )
    height: int = Field(
        default=512, description="The height of the generated image.", gt=0, le=4096
    )


ImageSizePreset = Literal[
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]


ImageSizeInput = Union[ImageSize, ImageSizePreset]


class LoraWeight(BaseModel):
    path: str = Field(
        description="URL or the path to the LoRA weights.",
        examples=[
            "https://civitai.com/api/download/models/135931",
            "https://filebin.net/3chfqasxpqu21y8n/my-custom-lora-v1.safetensors",
        ],
    )
    scale: float = Field(
        default=1.0,
        description="""
            The scale of the LoRA weight. This is used to scale the LoRA weight
            before merging it with the base model.
        """,
        ge=0.0,
        le=1.0,
    )
