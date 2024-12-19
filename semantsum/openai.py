from typing import Optional, Union, Any

from classconfig import ConfigurableValue, ConfigurableFactory
from classconfig.validators import StringValidator
from json_repair import json_repair
from openai import OpenAI

from semantsum.api import API
from semantsum.prompt_builder import PromptBuilder
from semantsum.summarizer import SingleDocSummarizer, Summarizer, QueryBasedMultiDocSummarizer


class OpenAIAPI(API):
    """
    Handles requests to the API.
    """

    base_url: Optional[str] = ConfigurableValue(desc="Base URL for OpenAI API.", user_default=None, voluntary=True)

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)


class OpenAIWithPromptBuilder(Summarizer):
    """
    Base class for OpenAI summarizers that use prompt builder to allow user to define configurable prompt.
    """
    api: OpenAIAPI = ConfigurableFactory(OpenAIAPI, "OpenAI API configuration.")
    model: str = ConfigurableValue("Name of model that should be used.", user_default="gpt-4o-mini")
    prompt_builder: PromptBuilder = ConfigurableFactory(PromptBuilder, "Prompt builder.")
    response_format: Optional[dict] = ConfigurableValue("Format of the response.", voluntary=True,
                                                        user_default=None)

    def __init__(self, api: OpenAIAPI, model: str, prompt_builder: PromptBuilder,
                 response_format: Optional[dict] = None):
        self.api = api
        self.model = model
        self.prompt_builder = prompt_builder
        self.response_format = response_format

    def __call__(self, template_fields: dict) -> Union[str, dict[str, Any]]:
        template = self.prompt_builder.build(template_fields)

        if isinstance(template, str):
            # segmented string or plain string
            res = self.api.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": template}
                ],
                response_format=self.response_format
            ).choices[0].message.content
        else:
            res = self.api.client.chat.completions.create(
                model=self.model,
                messages=template,
                response_format=self.response_format
            ).choices[0].message.content

        if self.response_format is not None and self.response_format["type"] == "json_schema":
            return json_repair.loads(res)
        else:
            return res


class OpenAISingleDocSummarizer(SingleDocSummarizer, OpenAIWithPromptBuilder):
    prompt_builder: PromptBuilder = ConfigurableFactory(PromptBuilder,
                                                        "Prompt builder. Available fields are: doc.")

    def __call__(self, doc: str) -> Union[str, dict[str, Any]]:
        return OpenAIWithPromptBuilder.__call__(self, {"doc": doc})


class OpenAIQueryBasedMultiDocSummarizer(QueryBasedMultiDocSummarizer, OpenAIWithPromptBuilder):
    prompt_builder: PromptBuilder = ConfigurableFactory(PromptBuilder,
                                                        "Prompt builder. Available fields are: query, docs.")

    def __call__(self, query: str, docs: list[str]) -> Union[str, dict[str, Any]]:
        return OpenAIWithPromptBuilder.__call__(self, {"query": query, "docs": docs})

