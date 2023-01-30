from pydantic import BaseModel, Field


class NEROutput(BaseModel):
    label: str = Field(None, alias="entity_group")
    score: float
    word: str
    start: int = None
    end: int = None

    class Config:
        extra = "allow"
