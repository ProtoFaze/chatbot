from pydantic  import BaseModel
from typing import Literal

class Intent(BaseModel):
    reasoning: str
    intent: Literal["normal","register","rag","verify","abuse"]


class AnalysisResults(BaseModel):
    overall_user_sentiment: Literal['negative', 'neutral', 'positive']
    interest_in_product: Literal['low', 'low-medium', 'medium', 'medium-high', 'high'] 
    key_topics: list[str]
    is_user_not_interested_in_signing_up: bool
    disengagement_reason: str