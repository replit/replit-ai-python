from pydantic import BaseModel


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
