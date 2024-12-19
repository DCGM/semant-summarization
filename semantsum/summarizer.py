from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Sequence


class Summarizer(ABC):
    """
    Abstract base class for summarizers.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Summarizes the input.

        :return: Summary.
            str: summary in form of plain text
            dict: structured summary
        """
        ...


class SingleDocSummarizer(Summarizer, ABC):
    """
    Abstract base class for summarizers that summarize single document.
    """

    @abstractmethod
    def __call__(self, inp: str) -> Union[str, Dict[str, Any]]:
        """
        Summarizes the input.

        :param inp: Input data.
        :return: Summary.
            str: summary in form of plain text
            dict: structured summary
        """
        ...


class QueryBasedMultiDocSummarizer(Summarizer, ABC):
    """
    Abstract base class for summarizers that summarize multiple documents based on query.
    """

    @abstractmethod
    def __call__(self, query: str, docs: Sequence[str]) -> Union[str, Dict[str, Any]]:
        """
        Summarizes the input.

        :param query: Query.
        :param docs: Documents.
        :return: Summary.
            str: summary in form of plain text
            dict: structured summary
        """
        ...


