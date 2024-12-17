from pydantic import BaseModel, Field
from typing import Literal, Annotated


class AnalysisResults(BaseModel):
    overall_sentiment: Literal['positive', 'negative', 'neutral']  # Maintain simplicity here.
    interest_in_product: int = Field(ge=1, le=10, description='level of interest in the insurance product from a scale of 1 to 10')  # Ensures values fall within a valid range.
    intent_classification_accuracy: float = Field(ge=0.0, le=1.0, description='Classification accuracy of intent in percentage')  # Enforces percentage format.

    # Optional additions:
    key_topics: list[str] = Field(default_factory=list, description="Extracted key themes.")
    conversation_turns: int = Field(..., ge=1, description="Total number of turns in the conversation.")
    signup_disengagement_flag: bool = Field(default=False, description="Indicates whether the user is not interested in signing up.")
    signup_disengagement_reason: Annotated[str, Field(default=None, description="Reason for losing interest in signing up.")]