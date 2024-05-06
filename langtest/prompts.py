from typing import List, Union

from pydantic import BaseModel, Extra, validator


class MessageType(BaseModel):
    __field_order: List[str] = [
        "content",
        "context",
        "question",
        "original",
        "testcase",
        "options",
        "answer",
    ]

    class Config:
        extra = (
            Extra.allow
        )  # Allow any additional fields that are not explicitly declared

    @validator("*", pre=True, allow_reuse=True)
    def add_field(cls, v, values, field, **kwargs):
        if "fields" not in values:
            values["fields"] = []
        values["fields"].append(field)
        return v

    @property
    def get_template(self):
        """Generate a template string based on the dynamic fields of the instance."""

        temp = []
        for field in self.__field_order:
            if field in self.__dict__:
                temp.append(f"{field.title()}: {{{field}}}")
        return "\n" + "\n".join(temp)

    @property
    def get_example(self):
        """Generate an example string based on the dynamic fields of the instance."""
        # return {k: v for k, v in self.__dict__.items() if k != 'fields'}
        temp = {}
        for field in self.__field_order:
            if field in self.__dict__:
                temp[field] = self.__dict__[field]
        return temp

    @property
    def input_variables(self):
        temp = []
        for field in self.__field_order:
            if field in self.__dict__:
                temp.append(field)
        return temp


class Conversion(BaseModel):
    """Conversion model for the conversion of the input and output of the model."""

    user: MessageType
    ai: MessageType

    class Config:
        extra = (
            Extra.allow
        )  # Allow any additional fields that are not explicitly declared

    @validator("*", pre=True, allow_reuse=True)
    def add_field(cls, v, values, field, **kwargs):
        if "fields" not in values:
            values["fields"] = []
        values["fields"].append(field)
        return v

    @property
    def get_examples(self):
        """Generate a list of examples based on the dynamic fields of the instance."""
        return {**self.user.get_example, **self.ai.get_example}


class PromptConfig(BaseModel):
    instructions: str
    prompt_type: str
    examples: Union[Conversion, List[Conversion]] = None

    @property
    def get_examples(self):
        """Generate a list of examples based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return [self.examples.get_examples]
        elif isinstance(self.examples, list):
            return [example.get_examples for example in self.examples]
        return self.examples.get_examples

    @property
    def get_template(self):
        """Generate a template string based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return self.examples.user.get_template
        elif isinstance(self.examples, list):
            return [
                ("human", self.examples[0].user.get_template),
                ("ai", self.examples[0].ai.get_template),
            ]
        # return self.examples.get_template

    @property
    def get_input_variables(self):
        """Generate a list of input variables based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return self.examples.user.input_variables
        elif isinstance(self.examples, list):
            return self.examples[0].user.input_variables

    def prompt_style(self):
        """Generate a prompt based on the prompt type."""
        if self.prompt_type == "chat":
            from langchain.prompts import (
                ChatPromptTemplate,
                FewShotChatMessagePromptTemplate,
            )

            example_prompt = ChatPromptTemplate.from_messages(self.get_template)

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples=self.get_examples,
                example_prompt=example_prompt,
            )

            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.instructions),
                    few_shot_prompt,
                    # ('human', conf),
                    self.get_template[0],
                ]
            )
            return final_prompt

        elif self.prompt_type == "instruct":
            from langchain.prompts import FewShotPromptTemplate, PromptTemplate

            example = PromptTemplate.from_template(self.get_template)

            final_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_selector=example,
            )

    def get_prompt(self):
        return self.prompt_style()
