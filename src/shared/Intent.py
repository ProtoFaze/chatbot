from pydantic  import BaseModel
from typing import Literal

class IntentModel(BaseModel):
    reasoning: str
    intent: Literal["normal","register","rag","verify"]