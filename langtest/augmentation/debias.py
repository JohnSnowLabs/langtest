from typing import Dict, List, Literal, TypeVar, Union, Type
from pydantic import BaseModel, Field
import pandas as pd

_Schema = TypeVar("_Schema", bound=BaseModel)

_STANDARD_BIAS_EVALUATION_PROMPT = """
The following text contains biased information, such as discrimination and stereotyping,
which fall into various categories. Each category includes specific subcategories
separated by commas, as listed below:

Category: Sub-category1, Sub-category2, Sub-category3, ...

- Demographic Bias: Gender-specific, racial, ethnic, religious, age-related.
- Social Bias: Socio-economic, educational, occupational, geographical.
- Historical Bias: Cultural, traditional, colonial.
- Confirmation Bias: Selective observation, cherry-picking evidence.
- Evaluation Bias: Subjective judgment, misrepresentation in assessment.
- Aggregation Bias: Overgeneralization, stereotyping, data grouping errors.
- Algorithmic Bias: Model architecture, optimization processes, unfair weighting.
- Data Bias: Imbalanced datasets, exclusion of minority groups, labeling inaccuracies.
- Automation Bias: Overreliance on machine-generated outputs, neglecting human oversight.

Please identify the category and subcategories of bias present in the text, and provide a
step-by-step de-biasing version with justifications for each adjustment.
"""


class _BiasDetectionRequest(BaseModel):
    """
    BiasDetectionRequest model.

    Attributes:
        text (str): Text to detect bias.
    """

    text: str = Field(..., description="Text to detect bias")


class _BiasDetectionResponse(BaseModel):
    """
    Represents the response structure for bias detection results.

    Attributes:
        category (str): Category of bias.
        sub_category (str): Sub-category of bias.
        bias_rationale (str): Reason for bias.
    """

    class Step(BaseModel):
        biased_word: str
        debiased_word: str

        def __repr__(self):
            return f"{self.biased_word} -> {self.debiased_word};"

        def __str__(self):
            return f"{self.biased_word} -> {self.debiased_word};"

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
    sub_category: str = Field(..., description="Sub-category of bias")
    bias_rationale: str = Field(..., description="Reason for bias")
    steps: List[Step] = Field(
        ...,
        description="Simple Steps to mitigate bias by changing the words or phrases in the text",
    )


class _TextDebiasingRequest(BaseModel):
    """
    Model for requesting text debiasing operations.

    Attributes:
        text (str): Text to debias.
    """

    text: str = Field(..., description="Text to debias")


class _DebiasedTextResponse(BaseModel):
    """
    Represents a debiased text response.

    Attributes:
        debiased_text (str): The debiased version of the text.
    """

    debiased_text: str = Field(..., description="Debiased text")


class _DebiasingRequest(BaseModel):
    """Request model for debiasing operations.

    Attributes:
        dataset (pd.DataFrame): Dataset to debias.
        text_column (str): Column name containing text.
    """

    dataset: pd.DataFrame = Field(..., description="Dataset to debias")
    text_column: str = Field(..., description="Column name containing text")

    model_config: Dict = {
        "arbitrary_types_allowed": True,
    }


class _DebiasingResult(BaseModel):
    """
    Representation of the debiasing result.

    Attributes:
        debiased_dataset (pd.DataFrame): Debiased dataset.
        debias_info (List[Dict]): Information about debiasing.
    """

    debiased_dataset: pd.DataFrame = Field(..., description="Debiased dataset")
    debias_info: List[Dict] = Field(..., description="Information about debiasing")

    model_config: Dict = {
        "arbitrary_types_allowed": True,
    }


class DebiasTextProcessing:
    def __init__(
        self,
        model: str,
        hub: str,
        system_prompt: str = _STANDARD_BIAS_EVALUATION_PROMPT,
        model_kwargs: Dict = None,
    ):
        self.model = model
        self.hub = hub
        self.system_prompt = system_prompt
        self.model_kwargs = model_kwargs
        self.debias_info = pd.DataFrame(
            columns=[
                "row",
                "original text",
                "reason",
                "category",
                "sub_category",
                "steps",
            ]
        )

    def initialize(
        self, input_dataset: pd.DataFrame, text_column: str, output_dataset: str = None
    ):
        self.input_dataset = input_dataset
        self.text_column = text_column
        self.output_dataset: pd.DataFrame = output_dataset

        if output_dataset is None:
            self.output_dataset = pd.DataFrame(columns=["biased_text", "debiased_text"])

        # reset debias_info
        self.debias_info = pd.DataFrame(
            columns=[
                "row",
                "original text",
                "reason",
                "category",
                "sub_category",
                "steps",
            ]
        )

    def identify_bias(self):
        for index, row in self.input_dataset.iterrows():
            text = row[self.text_column]
            category, sub_category, rationale, steps = self.detect_bias(text)
            if rationale:
                if index not in self.debias_info["row"].values:
                    self.debias_info.loc[len(self.debias_info)] = {
                        "row": index,
                        "original text": text,
                        "reason": rationale,
                        "category": category,
                        "sub_category": sub_category,
                        "steps": steps,
                    }
                else:
                    self.debias_info.loc[row["row"], "original text"] = text
                    self.debias_info.loc[row["row"], "reason"] = rationale
                    self.debias_info.loc[row["row"], "category"] = category
                    self.debias_info.loc[row["row"], "sub_category"] = sub_category
                    self.debias_info.loc[row["row"], "steps"] = steps
                self.debias_info = self.debias_info.reset_index(drop=True)

    def detect_bias(
        self, text: Union[str, _BiasDetectionRequest]
    ) -> _BiasDetectionResponse:
        # Placeholder for bias detection logic
        if isinstance(text, _BiasDetectionRequest):
            text = text.text

        output_data = self.interaction_llm(
            text, output_schema=_BiasDetectionResponse, system_prompt=self.system_prompt
        )

        return (
            output_data.category,
            output_data.sub_category,
            output_data.bias_rationale,
            output_data.steps,
        )

    def interaction_llm(
        self, text: str, output_schema: Type[_Schema], system_prompt: str
    ) -> _Schema:

        if self.hub == "openai":
            output_data = self.get_openai(
                text, system_prompt, output_schema, self.model_kwargs
            )
        elif self.hub == "ollama":
            output_data = self.get_ollama(
                text, system_prompt, output_schema, self.model_kwargs
            )

        return output_data

    def debias_text(
        self, text: str, category: str, sub_category: str, reason: str, steps: List[str]
    ) -> str:
        # Placeholder for debiasing logic
        step_by_step = "\n".join(f"- {str(step)}" for step in steps)
        system_prompt = (
            f"Debias the text with the following bias information and reason.\n\n"
            f"Category: {category}\n"
            f"Sub-category: {sub_category}\n"
            f"Reason: {reason}\n\n"
            f"Step by Step Debiasing Instructions:\n{step_by_step}"
        )

        prompt = f"""The following text contains bias. Please rewrite it to eliminate bias by rephrasing or modifying specific words and phrases.\nText: {text}"""

        debiased_text = self.interaction_llm(
            prompt, output_schema=_DebiasedTextResponse, system_prompt=system_prompt
        )

        return debiased_text.debiased_text

    def apply_debiasing(self):
        for idx, row in self.debias_info.iterrows():
            original_text = self.input_dataset.at[row["row"], self.text_column]
            debiased_text = self.debias_text(
                original_text,
                category=row["category"],
                sub_category=row["sub_category"],
                reason=row["reason"],
                steps=row["steps"],
            )
            self.output_dataset.loc[row["row"], "biased_text"] = original_text
            self.output_dataset.loc[row["row"], "debiased_text"] = debiased_text

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
        self, text, system_prompt, output_schema: Type[_Schema], *args, **kwargs
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
        self, text, system_prompt, output_schema: Type[_Schema], model_kwargs: Dict = None
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
