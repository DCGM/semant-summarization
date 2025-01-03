from abc import ABC, abstractmethod
from typing import Sequence, Optional

from classconfig import ConfigurableValue, ConfigurableSubclassFactory
from classconfig.validators import IntegerValidator, StringValidator

from semantsum.api import API


class Summarizer(ABC):
    """
    Abstract base class for summarizers.
    """

    def check_arguments(self, text: Sequence[str], query: Optional[str] = None):
        """
        Checks arguments before summarization.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :raise ValueError: if the input is invalid
        """
        ...

    @abstractmethod
    def summ_str(self, text: Sequence[str], query: Optional[str] = None) -> str:
        """
        Summarizes the input and returns it as a string.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :return: Summary.
        """
        ...

    def summ_struct(self, text: Sequence[str], query: Optional[str] = None) -> Sequence[tuple[Optional[str], str]]:
        """
        Summarizes the input and returns it as a structured summary.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :return: Structured summary in form of sequence of tuples with two string elements.
            The first element is reserved for label and the second for content.
            If structured summary is not supported, returns list with one tuple with None label and summary as content.
        """
        return [
            (None, self.summ_str(text, query))
        ]


class APIBasedSummarizer(Summarizer, ABC):
    """
    Abstract base class for summarizers that use API.
    """

    api: API = ConfigurableSubclassFactory(API, "API configuration.")


class SummarizerWorkflow:
    """
    Summarization workflow is mainly used for configuration purposes.
    """

    name: str = ConfigurableValue("Name of the summarization.", validator=StringValidator())
    summary_type: str = ConfigurableValue("Summarization type.", validator=StringValidator())
    provider: str = ConfigurableValue("Provider of the summarization.", validator=StringValidator())
    description: str = ConfigurableValue("Description of the summarization.", validator=StringValidator())
    version: str = ConfigurableValue("Version of the summarization workflow.", validator=StringValidator())
    recommended_num_texts: int = ConfigurableValue("Recommended number of texts for summarization.", validator=IntegerValidator())
    summarizer: Summarizer = ConfigurableSubclassFactory(Summarizer, "Summarizer configuration.")

    def __init__(self, name: str, summary_type: str, provider: str,
                 description: str, version: str, summarizer: Summarizer, recommended_num_texts: int):
        self.name = name
        self.summary_type = summary_type
        self.provider = provider
        self.description = description
        self.version = version
        self.summarizer = summarizer
        self.recommended_num_texts = recommended_num_texts

    def summ_str(self, text: Sequence[str], query: Optional[str] = None) -> str:
        """
        Summarizes the input and returns it as a string.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :return: Summary.
        """
        return self.summarizer.summ_str(text, query)

    def summ_struct(self, text: Sequence[str], query: Optional[str] = None) -> Sequence[tuple[Optional[str], str]]:
        """
        Summarizes the input and returns it as a structured summary.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :return: Structured summary in form of sequence of tuples with two string elements.
            The first element is reserved for label and the second for content.
            If structured summary is not supported, returns list with one tuple with None label and summary as content.
        """
        return self.summarizer.summ_struct(text, query)


class APIBasedSummarizerWorkflow(SummarizerWorkflow):
    """
    Summarization workflow that uses APIBasedSummarizer.
    """

    summarizer: APIBasedSummarizer = ConfigurableSubclassFactory(APIBasedSummarizer, "Summarizer configuration.")


class SingleDocSummarizer(Summarizer, ABC):
    """
    Abstract base class for summarizers that summarize single document.
    """
    TYPE = "single document"

    def check_arguments(self, text: Sequence[str], query: Optional[str] = None):
        """
        Checks arguments before summarization.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :raise ValueError: if the input is invalid
        """
        if len(text) != 1:
            raise ValueError(f"Single document summarizer expects exactly one text, but {len(text)} were provided.")

        if query is not None:
            raise ValueError("Single document summarizer does not support queries.")


class QueryBasedMultiDocSummarizer(Summarizer, ABC):
    """
    Abstract base class for summarizers that summarize multiple documents based on query.
    """
    TYPE = "query-based multi-document"

    def check_arguments(self, text: Sequence[str], query: Optional[str] = None):
        """
        Checks arguments before summarization.

        :param text: input in form of sequence of strings.
        :param query: optional query string
        :raise ValueError: if the input is invalid
        """
        if len(text) < 1:
            raise ValueError(f"Query-based multi-document summarizer expects at least one text.")

        if query is None:
            raise ValueError("Query-based multi-document summarizer requires a query.")
