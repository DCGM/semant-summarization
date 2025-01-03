from typing import Optional, Union, Sequence

import jinja2
from classconfig import ConfigurableValue, ConfigurableFactory, ConfigurableMixin
from classconfig.validators import StringValidator, AnyValidator, IsNoneValidator
from json_repair import json_repair
from openai import OpenAI

from semantsum.api import API
from semantsum.prompt_builder import PromptBuilder
from semantsum.summarizer import SingleDocSummarizer, QueryBasedMultiDocSummarizer, APIBasedSummarizer


class OpenAIAPI(API):
    """
    Handles requests to the API.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)


class StructuredSchema(ConfigurableMixin):
    """
    Defines structured summary.
    """

    summary_name: str = ConfigurableValue("Name of the summary field. (e.g., timeline)", user_default="summary", validator=StringValidator())
    summary_description: Optional[str] = ConfigurableValue("Description of the summary field.", voluntary=True, user_default=None, validator=AnyValidator([StringValidator(), IsNoneValidator()]))
    sequence_name: str = ConfigurableValue("Name of the sequence field. (e.g., events)", user_default="sequence", validator=StringValidator())
    sequence_description: Optional[str] = ConfigurableValue("Description of the sequence field.", voluntary=True, user_default=None, validator=AnyValidator([StringValidator(), IsNoneValidator()]))
    label_name: str = ConfigurableValue("Name of the label field. (e.g., time_descriptor)", user_default="label", validator=StringValidator())
    label_description: Optional[str] = ConfigurableValue("Description of the label field.", voluntary=True, user_default=None, validator=AnyValidator([StringValidator(), IsNoneValidator()]))
    content_name: str = ConfigurableValue("Name of the content field. (e.g., summary)", user_default="content", validator=StringValidator())
    content_description: Optional[str] = ConfigurableValue("Description of the content field.", voluntary=True, user_default=None, validator=AnyValidator([StringValidator(), IsNoneValidator()]))

    def generate_json_schema(self) -> dict:
        """
        Generates JSON schema for the structured summary.

        :return: JSON schema
        """

        res = {
          "type": "json_schema",
          "json_schema": {
            "name": self.summary_name,
            "strict": True,
            "schema": {
              "type": "object",
              "properties": {
                self.sequence_name: {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      self.label_name: {
                        "type": "string"
                      },
                      self.content_name: {
                        "type": "string"
                      }
                    },
                    "required": [self.label_name, self.content_name],
                    "additionalProperties": False
                  }
                }
              },
              "required": [self.sequence_name],
              "additionalProperties": False
            },
          }
        }

        if self.summary_description is not None:
            res["json_schema"]["description"] = self.summary_description

        if self.sequence_description is not None:
            res["json_schema"]["schema"]["properties"][self.sequence_name]["description"] = self.sequence_description

        if self.label_description is not None:
            res["json_schema"]["schema"]["properties"][self.sequence_name]["items"]["properties"][self.label_name]["description"] = self.label_description

        if self.content_description is not None:
            res["json_schema"]["schema"]["properties"][self.sequence_name]["items"]["properties"][self.content_name]["description"] = self.content_description

        return res

    def parse_response(self, response: str) -> list[tuple[Optional[str], str]]:
        """
        Parses response from the API.

        :param response: Response from the API.
        :return: Structured summary.
        """

        response = json_repair.loads(response)

        return [
            (item[self.label_name], item[self.content_name])
            for item in response[self.sequence_name]
        ]


class OpenAIWithPromptBuilder(APIBasedSummarizer):
    """
    Base class for OpenAI summarizers that use prompt builder to allow user to define configurable prompt.
    """

    api: OpenAIAPI = ConfigurableFactory(OpenAIAPI, "OpenAI API configuration.")
    model: str = ConfigurableValue("Name of model that should be used.", user_default="gpt-4o-mini")
    prompt_builder: PromptBuilder = ConfigurableFactory(PromptBuilder, "Prompt builder. Available fields: text, query. Text is list of strings and query is optional string (None when missing).")
    structured: Optional[StructuredSchema] = ConfigurableFactory(StructuredSchema, "Structured summary configuration.", voluntary=True)
    structured_2_str: str = ConfigurableValue("Jinja template for converting item of structured summary to string (available fields: label, content).",
                                              user_default="{{label}}: {{content}}", validator=StringValidator())

    def __init__(self, api: OpenAIAPI, model: str, prompt_builder: PromptBuilder,
                 structured: Optional[StructuredSchema] = None, structured_2_str: str = "{label}: {content}"):
        self.api = api
        self.model = model
        self.prompt_builder = prompt_builder
        self.structured = structured
        self.jinja = jinja2.Environment()
        self.structured_2_str = self.jinja.from_string(structured_2_str)

    def __call__(self, template_fields: dict) -> Union[str, list[tuple[Optional[str], str]]]:
        """
        Calls the summarizer.

        :param template_fields: Fields for the prompt builder.
        :return: summary or structured summary
        """
        template = self.prompt_builder.build(template_fields)

        response_format = None
        if self.structured is not None:
            response_format = self.structured.generate_json_schema()

        if isinstance(template, str):
            # segmented string or plain string
            res = self.api.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": template}
                ],
                response_format=response_format
            ).choices[0].message.content
        else:
            res = self.api.client.chat.completions.create(
                model=self.model,
                messages=template,
                response_format=response_format
            ).choices[0].message.content

        if self.structured is not None:
            return self.structured.parse_response(res)
        return res

    def convert_structured_2_str(self, structured: list[tuple[Optional[str], str]]) -> str:
        """
        Converts structured summary to a string.

        :param structured: Structured summary.
        :return: summary
        """

        return "\n".join(
            self.structured_2_str.render({"label": item[0], "content": item[1]})
            for item in structured
        )

    def summ_str(self, text: Sequence[str], query: Optional[str] = None) -> str:
        self.check_arguments(text, query)
        res = self({"text": text, "query": query})
        if not isinstance(res, str):
            return self.convert_structured_2_str(res)
        return res

    def summ_struct(self, text: Sequence[str], query: Optional[str] = None) -> Sequence[tuple[Optional[str], str]]:
        self.check_arguments(text, query)
        res = self({"text": text, "query": query})
        if self.structured is None:
            return [(None, res)]

        return res


class OpenAISingleDocSummarizer(SingleDocSummarizer, OpenAIWithPromptBuilder):
    ...


class OpenAIQueryBasedMultiDocSummarizer(QueryBasedMultiDocSummarizer, OpenAIWithPromptBuilder):
    ...
