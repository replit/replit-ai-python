from typing import Optional

from pydantic import BaseModel


class TokenCountMetadata(BaseModel):
    billableTokens: int = 0
    unbilledTokens: int = 0
    billableCharacters: int = 0
    unbilledCharacters: int = 0


class GoogleMetadata(BaseModel):
    inputTokenCount: Optional[TokenCountMetadata] = None
    outputTokenCount: Optional[TokenCountMetadata] = None


class GoogleEmbeddingMetadata(BaseModel):
    tokenCountMetadata: Optional[TokenCountMetadata] = None
