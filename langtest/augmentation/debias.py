from typing import Dict, List, Literal, Union
from pydantic import BaseModel, Field
import pandas as pd


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


class DebiasingResult(BaseModel):
    """
    Representation of the debiasing result.

    Attributes:
        debiased_dataset (pd.DataFrame): Debiased dataset.
        debias_info (List[Dict]): Information about debiasing.
    """

    debiased_dataset: pd.DataFrame = Field(..., title="Debiased dataset")
    debias_info: List[Dict] = Field(..., title="Information about debiasing")


class DebiasTextProcessing:
    def __init__(self, dataset: pd.DataFrame, text_column: str):
        self.dataset = dataset
        self.text_column = text_column
        self.debias_info = []

    def initialize(self, model: str, hub: str):
        # Placeholder for model initialization
        self.debias_model = (model, hub)

    def identify_bias(self):
        for index, row in self.dataset.iterrows():
            text = row[self.text_column]
            reason, category, sub_category = self.detect_bias(text)
            if reason:
                self.debias_info.append(
                    {
                        "row": index,
                        "reason": reason,
                        "category": category,
                        "sub_category": sub_category,
                    }
                )

    def detect_bias(
        self, text: Union[str, BiasDetectionRequest]
    ) -> BiasDetectionResponse:
        # Placeholder for bias detection logic

        return (None, None, None)

    def debias_text(self, text: str):
        # Placeholder for debiasing logic
        return text

    def apply_debiasing(self):
        for info in self.debias_info:
            original_text = self.dataset.at[info["row"], self.text_column]
            debiased_text = self.debias_text(original_text)
            self.dataset.at[info["row"], self.text_column] = debiased_text

    def process(self):
        self.identify_bias()
        self.apply_debiasing()
        return self.dataset, self.debias_info

    def load_data(self, source: str, source_type: str):
        if source_type == "csv":
            self.dataset = pd.read_csv(source)
        elif source_type == "json":
            self.dataset = pd.read_json(source)
        elif source_type == "excel":
            self.dataset = pd.read_excel(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
