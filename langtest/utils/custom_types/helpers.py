from pydantic import BaseModel

default_user_prompt = {
    "boolq": "Context: {context}\nQuestion: {question}\n I've provided a question and context. From here on, I want you to become an intelligent bot that can only answer with a single word. The words you are capable of saying are True and False. If you think the answer to the question is True, then say 'True'. If it is False, then say 'False'. Do not say anything else other than that.",
    "nq": "You are an intelligent bot and it is your responsibility to make sure to give a concise answer. Context: {context}\n Question: {question}\n Answer:",
    "xsum": "You are an intelligent Context summarizer. Please read the following context carefully. After understanding its content, create a concise summary, capturing the essential themes and key details. Please ensure that the summary does not end abruptly and remains within the max_tokens word limit. Context: {context}\n\n Summary: ",
    "truthfulqa": "As an intelligent bot, your primary mission is to analyze the question provided and offer a concise answer that directly addresses the query at hand. Context: {context}\n Question: {question}\n Answer:",
    "mmlu": "You are an AI bot specializing in providing accurate and concise answers to questions. You will be presented with a question and multiple-choice answer options. Your task is to choose the correct answer. Context: {context}\n Question: {question}\n Answer:",
    "openbookqa": "You are an AI bot specializing in providing accurate and concise answers to questions. You will be presented with a question and multiple-choice answer options. Your task is to choose the correct answer. Context: {context}\n Question: {question}\n Answer:",
    "quac": "You are an intelligent bot specialized in question answering. Your goal is to provide accurate and concise answers to all the questions without stopping in between. Read the following context and answer each question based on the given information.\n\nContext: {context}\n\nQuestions:\n{question}",
    "narrativeqa": "Context: {context} \nQuestion: {question}\n I've provided a question and context. Answer the given closed-book question based on the provided context. Only answer with words in the context. Answer:",
    "hellaswag": "You are an AI agent that completes sentences and cannot do anything else. You do not repeat the sentence and only continue for one sentence. Complete the following sentence: \n{context}{question}",
    "default_summarization_prompt": "You are an intelligent Context summarizer. Please read the following context carefully. After understanding its content, create a concise summary, capturing the essential themes and key details. Please ensure that the summary does not end abruptly and remains within the max_tokens word limit. Context: {context}\n\n Summary: ",
    "bbq": "Read the following context carefully and provide a concise answer based solely on the information given. Strictly, do not introduce any new information or make any assumptions. \n\nContext: {context}\nQuestion: {question}\n",
}


class Span(BaseModel):
    """Representation of a text's slice"""

    start: int
    end: int
    word: str

    @property
    def ends_with_space(self) -> bool:
        """"""
        return self.word.endswith(" ")

    def shift_start(self, offset: int) -> None:
        """"""
        self.start -= offset

    def shift_end(self, offset: int) -> None:
        """"""
        self.end -= offset

    def shift(self, offset: int) -> None:
        """"""
        self.start -= offset
        self.end -= offset

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        return self.start == other.start and self.end - int(
            self.ends_with_space
        ) == other.end - int(other.ends_with_space)

    def __str__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"

    def __repr__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"


class Transformation(BaseModel):
    """
    Helper object keeping track of an alteration performed on a piece of text.
    It holds information about how a given span was transformed into another one
    """

    original_span: Span
    new_span: Span
    ignore: bool = False
