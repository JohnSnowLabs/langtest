from pydantic import BaseModel, Field
from typing import List


class NEROutput(BaseModel):
    entity: str = Field(None, alias="entity_group")
    word: str
    start: int
    end: int
    score: float = None

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True


class SequenceLabel(BaseModel):
    label: str
    score: float


class SequenceClassificationOutput(BaseModel):
    text: str
    labels: List[SequenceLabel]

    def __str__(self):
        labels = {elt.label: elt.score for elt in self.labels}
        return f"SequenceClassificationOutput(text='{self.text}', labels={labels})"
