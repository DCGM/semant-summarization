from unittest import TestCase

from segmentedstring import SegmentedString

from semantsum.prompt_builder import PromptBuilder


class TestPromptBuilder(TestCase):
    def test_plain_text_prompt(self):
        builder = PromptBuilder("Hello my name is {{ name }}.")
        prompt = builder.build({"name": "John"})

        self.assertEqual(prompt, "Hello my name is John.")

    def test_dict_prompt(self):
        builder = PromptBuilder({
            "greeting": "Hello my name is {{ name }}. ",
            "farewell": "Goodbye {{ name }}."
        })

        prompt = builder.build({"name": "John"})

        self.assertIsInstance(prompt, SegmentedString)
        self.assertEqual("Hello my name is John. Goodbye John.", prompt)
        self.assertSequenceEqual(["greeting", "farewell"], prompt.labels)
        self.assertSequenceEqual(["Hello my name is John. ", "Goodbye John."], prompt.segments)

    def test_role_content_prompt(self):
        builder = PromptBuilder([
            {"role": "assistant", "content": "Hello my name is {{ name }}. "},
            {"role": "user", "content": "Goodbye {{ name }}."}
        ])

        prompt = builder.build({"name": "John"})

        self.assertIsInstance(prompt, list)

        self.assertEqual(2, len(prompt))

        self.assertEqual("assistant", prompt[0]["role"])
        self.assertEqual("Hello my name is John. ", prompt[0]["content"])

        self.assertEqual("user", prompt[1]["role"])
        self.assertEqual("Goodbye John.", prompt[1]["content"])
