from pydantic import BaseModel, Field


class NEROutput(BaseModel):
    entity: str = Field(None, alias="entity_group")
    word: str
    start: int
    end: int
    score: float = None

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True


#--------------------------------
# Kalyan



#--------------------------------
# Arshan


#---------------------------------
# Tarik