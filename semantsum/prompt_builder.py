from typing import Union, Dict, Any

import jinja2
from classconfig import ConfigurableValue
from segmentedstring import SegmentedString


class PromptBuilder:

    template: Union[str | dict[str, str] | list[dict[str, str]]] = ConfigurableValue(
        "Jinja2 template for prompt sequence. It can be a string, dictionary with keys 'segment_name' and 'template', or a sequence of messages with role and content. If you use dictionary all parts will be concatenated and SegmentedString will be used for the rendered result.",
        voluntary=True
    )

    role_key_form: str = ConfigurableValue("How the dict key would be named in built prompt sequence. Doesn't affect key name in template config.", user_default="role", voluntary=True)
    content_key_form: str = ConfigurableValue("How the dict key would be named in built prompt sequence. Doesn't affect key name in template config.", user_default="content", voluntary=True)

    def __init__(self, template: Union[str | dict[str, str] | list[dict[str, str]]], role_key_form: str = "role",
                 content_key_form: str = "content"):
        self.template = template
        self.jinja = jinja2.Environment()
        self.jinja_template = self.create_jinja_template(template)
        self.role_key_form = role_key_form
        self.content_key_form = content_key_form

    def create_jinja_template(self, template: Union[str, dict[str, str], list[dict[str, str]]]) -> Union[jinja2.Template, Dict[str, jinja2.Template], list[tuple[str, jinja2.Template]]]:
        """
        Creates jinja template from the template.

        :param template: template
        :return: jinja template
        """

        if isinstance(template, dict):
            return {
                k: self.jinja.from_string(v) for k, v in template.items()
            }
        elif isinstance(template, list):
            return [(message["role"], self.jinja.from_string(message["content"])) for message in template]
        else:
            return self.jinja.from_string(template)

    def build(self, data: dict[str, Any]) -> Union[str, SegmentedString, list[dict[str, str]]]:
        """
        Builds a prompt sequence from the data.

        :param data: data
        :return: prompt sequence
        """
        if isinstance(self.jinja_template, dict):
            return SegmentedString(
                [self.jinja_template[k].render(data) for k in self.jinja_template.keys()],
                self.jinja_template.keys()
            )
        elif isinstance(self.jinja_template, list):
            return [
                {
                    self.role_key_form: role,
                    self.content_key_form: template.render(data)
                } for role, template in self.jinja_template
            ]
        else:
            return self.jinja_template.render(data)
