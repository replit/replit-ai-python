from pydantic import BaseModel
from typing import Dict, Any, List


class GoogleCitation(BaseModel):
  startIndex: int
  endIndex: int
  url: int
  title: str
  license: str
  # Documented as "Possible formats are YYYY, YYYY-MM, YYYY-MM-DD."
  publicationDate: str


class GoogleCitationMetadata(BaseModel):
  citations: List[GoogleCitation] = []


class GoogleSafetyAttributes(BaseModel):
  blocked: bool = False
  categories: List[str] = []
  scores: List[float] = []


class GooglePredictionMetadata(BaseModel):
  safetyAttributes: GoogleSafetyAttributes
  citationMetadata: GoogleCitationMetadata


class TextGenerationResponse(BaseModel):
  """
  Class representing the response from text generation model.
  
  Attributes:
      is_blocked (bool): Flag indicating whether the output was blocked due to content safety filters.
      raw_prediction_response (Dict[str, Any]): Raw response from the AI model.
      safety_attributes (Dict[str, float]): Dictionary with safety attributes of the generated text.
      text (str): Generated text.
  """
  is_blocked: bool
  raw_prediction_response: Dict[str, Any]
  safety_attributes: Dict[str, float]
  text: str
