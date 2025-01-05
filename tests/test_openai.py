import unittest
from unittest.mock import patch, MagicMock

from semantsum.openai import OpenAIAPI, OpenAISingleDocSummarizer, OpenAIQueryBasedMultiDocSummarizer, \
    OpenAIWithPromptBuilder
from semantsum.prompt_builder import PromptBuilder


class TestOpenAIWithPromptBuilder(unittest.TestCase):

    def setUp(self):
        self.api = MagicMock(
            client=MagicMock(
                chat=MagicMock(
                    completions=MagicMock(
                        create=MagicMock(
                            return_value=MagicMock(
                                choices=[MagicMock(
                                    message=MagicMock(
                                        content="response"
                                    )
                                )]
                            )
                        )
                    )
                )
            )
        )

        self.prompt_builder = MagicMock(
            build=MagicMock(
                return_value="template"
            )
        )

    def test_convert_structured_2_str(self):
        summarizer = OpenAIWithPromptBuilder(
            api=MagicMock(),
            model="model",
            prompt_builder=MagicMock(),
            structured_2_str="{{label}} -> {{content}}"
        )

        res = summarizer.convert_structured_2_str([
            ("label1", "content1"),
            ("label2", "content2")
        ])

        self.assertEqual(res, "label1 -> content1\nlabel2 -> content2")

    def test_summ_str(self):
        summarizer = OpenAIWithPromptBuilder(
            api=self.api,
            model="model",
            prompt_builder=self.prompt_builder
        )
        res = summarizer.summ_str(["text"], "query")
        self.assertEqual("response", res)

        self.prompt_builder.build.assert_called_once_with({"text": ["text"], "query": "query"})
        self.api.client.chat.completions.create.assert_called_once_with(
            model="model",
            messages=[{"role": "user", "content": "template"}],
            response_format=None
        )

    def test_summ_struct_none(self):
        summarizer = OpenAIWithPromptBuilder(
            api=self.api,
            model="model",
            prompt_builder=self.prompt_builder
        )
        res = summarizer.summ_struct(["text"], "query")
        self.assertEqual([(None, "response")], res)

        self.prompt_builder.build.assert_called_once_with({"text": ["text"], "query": "query"})
        self.api.client.chat.completions.create.assert_called_once_with(
            model="model",
            messages=[{"role": "user", "content": "template"}],
            response_format=None
        )

    def test_summ_struct(self):
        summarizer = OpenAIWithPromptBuilder(
            api=self.api,
            model="model",
            prompt_builder=self.prompt_builder,
            structured=MagicMock(
                generate_json_schema=MagicMock(
                    return_value="json_schema"
                ),
                parse_response=MagicMock(
                    return_value=[("label", "response")]
                )
            )
        )
        res = summarizer.summ_struct(["text"], "query")
        self.assertSequenceEqual([("label", "response")], res)

        self.prompt_builder.build.assert_called_once_with({"text": ["text"], "query": "query"})
        self.api.client.chat.completions.create.assert_called_once_with(
            model="model",
            messages=[{"role": "user", "content": "template"}],
            response_format="json_schema"
        )


if __name__ == '__main__':
    unittest.main()
