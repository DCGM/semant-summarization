from pydantic import BaseModel
from typing import List, Tuple


class SummarizationConfig(BaseModel):
    name: str
    summary_type: str
    provider: str
    description: str
    version: str
    recommended_num_texts: int


class Response(BaseModel):
    status: str
    info: str


class SummarizationStringResponse(Response):
    summary: str


class SummaryItem(BaseModel):
    item: str | None
    text: str


class SummarizationStructResponse(Response):
    summary: list[SummaryItem]
