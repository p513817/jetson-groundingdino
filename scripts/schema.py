from typing import Optional

from pydantic import BaseModel


class DinoRequestBody(BaseModel):
    prompt: str
    image: str
    box_threshold: Optional[float] = 0.35
    text_threshold: Optional[float] = 0.25
