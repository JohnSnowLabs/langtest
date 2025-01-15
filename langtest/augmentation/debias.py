from typing import Dict, List, Literal, TypeVar, Union
from pydantic import BaseModel, Field
import pandas as pd

_Schema = TypeVar("_Schema", bound=BaseModel)


class BiasDetectionRequest(BaseModel):
    """
    BiasDetectionRequest model.

    Attributes:
        text (str): Text to detect bias.
    """

    text: str = Field(..., title="Text to detect bias")


class BiasDetectionResponse(BaseModel):
    """
    Represents the response structure for bias detection results.

    Attributes:
        category (str): Category of bias.
        sub_category (str): Sub-category of bias.
        bias_rationale (str): Reason for bias.
    """

    category: Literal[
        "demographic",
        "discrimination",
        "social",
        "historical",
        "confirmation",
        "evaluation",
        "aggregation",
        "algorithmic",
        "data",
        "automation",
    ] = Field(..., title="Category of bias")
    sub_category: str = Field(..., title="Sub-category of bias")
    bias_rationale: str = Field(..., title="Reason for bias")


class TextDebiasingRequest(BaseModel):
    """
    Model for requesting text debiasing operations.

    Attributes:
        text (str): Text to debias.
    """

    text: str = Field(..., title="Text to debias")


class DebiasedTextResponse(BaseModel):
    """
    Represents a debiased text response.

    Attributes:
        debiased_text (str): The debiased version of the text.
    """

    debiased_text: str = Field(..., title="Debiased text")


class DebiasingRequest(BaseModel):
    """Request model for debiasing operations.

    Attributes:
        dataset (pd.DataFrame): Dataset to debias.
        text_column (str): Column name containing text.
    """

    dataset: pd.DataFrame = Field(..., title="Dataset to debias")
    text_column: str = Field(..., title="Column name containing text")

    model_config: Dict = {
        "arbitrary_types_allowed": True,
    }


class DebiasingResult(BaseModel):
    """
    Representation of the debiasing result.

    Attributes:
        debiased_dataset (pd.DataFrame): Debiased dataset.
        debias_info (List[Dict]): Information about debiasing.
    """

    debiased_dataset: pd.DataFrame = Field(..., title="Debiased dataset")
    debias_info: List[Dict] = Field(..., title="Information about debiasing")

    model_config: Dict = {
        "arbitrary_types_allowed": True,
    }


class DebiasTextProcessing:
    def __init__(
        self, model: str, hub: str, system_prompt: str, model_kwargs: Dict = None
    ):
        self.model = model
        self.hub = hub
        self.system_prompt = system_prompt
        self.model_kwargs = model_kwargs
        self.debias_info = pd.DataFrame(
            {"row": [], "reason": [], "category": [], "sub_category": []}
        )

    def initialize(
        self, input_dataset: pd.DataFrame, text_column: str, output_dataset: str = None
    ):
        self.input_dataset = input_dataset
        self.text_column = text_column
        self.output_dataset: pd.DataFrame = output_dataset

    def identify_bias(self):
        for index, row in self.input_dataset.iterrows():
            text = row[self.text_column]
            category, sub_category, rationale = self.detect_bias(text)
            if rationale:
                if index not in self.debias_info["row"].values:
                    self.debias_info.loc[len(self.debias_info)] = {
                        "row": index,
                        "reason": rationale,
                        "category": category,
                        "sub_category": sub_category,
                    }
                else:
                    self.debias_info.loc[row["row"], "reason"] = rationale
                    self.debias_info.loc[row["row"], "category"] = category
                    self.debias_info.loc[row["row"], "sub_category"] = sub_category
                self.debias_info = self.debias_info.reset_index(drop=True)

    def detect_bias(
        self, text: Union[str, BiasDetectionRequest]
    ) -> BiasDetectionResponse:
        # Placeholder for bias detection logic
        if isinstance(text, BiasDetectionRequest):
            text = text.text

        output_data = self.interaction_llm(text, output_schema=BiasDetectionResponse)

        return (
            output_data.category,
            output_data.sub_category,
            output_data.bias_rationale,
        )

    def interaction_llm(self, text: str, output_schema: type[_Schema]) -> _Schema:

        if self.hub == "openai":
            output_data = self.get_openai(
                text, self.system_prompt, output_schema, self.model_kwargs
            )
        elif self.hub == "ollama":
            output_data = self.get_ollama(
                text, self.system_prompt, output_schema, self.model_kwargs
            )

        return output_data

    def debias_text(
        self, text: str, category: str, sub_category: str, reason: str
    ) -> str:
        # Placeholder for debiasing logic
        prompt = f"""
        Debias the text with following bias information and reason.

        Category: {category}
        Sub-category: {sub_category}
        Reason: {reason}

        Text: {text}

        Output:
        """

        debiased_text = self.interaction_llm(prompt, output_schema=DebiasedTextResponse)

        return debiased_text.debiased_text

    def apply_debiasing(self):
        for idx, row in self.debias_info.iterrows():
            original_text = self.input_dataset.at[row["row"], self.text_column]
            debiased_text = self.debias_text(
                original_text,
                category=row["category"],
                sub_category=row["sub_category"],
                reason=row["reason"],
            )
            self.output_dataset.loc[row["row"], self.text_column] = debiased_text

    def process(self):
        self.identify_bias()
        self.apply_debiasing()
        return self.output_dataset, self.debias_info

    def load_data(self, source: str, source_type: str):
        if source_type == "csv":
            self.dataset = pd.read_csv(source)
        elif source_type == "json":
            self.dataset = pd.read_json(source)
        elif source_type == "excel":
            self.dataset = pd.read_excel(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def get_openai(
        self, text, system_prompt, output_schema: type[_Schema], *args, **kwargs
    ) -> _Schema:
        import openai

        client = openai.Client()

        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=output_schema,
        )

        return response.choices[0].message.parsed

    def get_ollama(
        self, text, system_prompt, output_schema: type[_Schema], model_kwargs: Dict = None
    ) -> _Schema:
        from ollama import chat

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            format=output_schema.model_json_schema(),
            options=model_kwargs,
        )

        return output_schema.model_validate_json(
            response.get("message", {}).get("content")
        )
