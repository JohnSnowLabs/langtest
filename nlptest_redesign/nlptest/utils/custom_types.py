from pydantic import BaseModel, Field, validator


class NEROutput(BaseModel):
    entity: str = Field(None, alias="entity_group")
    word: str
    start: int
    end: int
    score: float = None

    @validator("entity")
    def normalize_entity(cls, entity):
        """"""
        mapper = {
            "PERSON": "PER",  # english spacy models
            "PS": "PER",  # korean spacy models
            "LC": "LOC",  # korean spacy models
            "OG": "ORG",  # korean spacy models
            "DT": "DATE",  # korean spacy models
            "TI": "TIME",  # korean spacy models
            "QT": "QUANTITY",  # korean spacy models,
            "EVN": "EVENT",
            "MSR": "",
            "OBJ": "PRODUCT",
            "PRS": "PER",
            "TME": "TIME",
            "WRK": "WORK_OF_ART"
        }
        for k, v in mapper.items():
            if k in entity:
                return entity.replace(k, v)
        return entity

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True
