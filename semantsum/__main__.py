from argparse import ArgumentParser

import inquirer
from classconfig import Config, ConfigurableFactory, ConfigurableSubclassFactory
from classconfig.classes import subclasses, sub_cls_from_its_name

from semantsum.api import API
from semantsum.openai import OpenAIWithPromptBuilder, OpenAIAPI
from semantsum.summarizer import Summarizer, QueryBasedMultiDocSummarizer, SingleDocSummarizer


class OpenAISummarizerWorkflow:
    summarizer: OpenAIWithPromptBuilder = ConfigurableSubclassFactory(OpenAIWithPromptBuilder,
                                                                      "Configuration for summarizer.")

    def __init__(self, summarizer: OpenAIWithPromptBuilder):
        self.summarizer = summarizer

    def __call__(self, *args, **kwargs):
        print(self.summarizer(*args, **kwargs))


def run_openai_summarizer(args):
    """
    Runs OpenAI summarizer.

    :param args: Parsed arguments.
    """

    # create summarizer from configuration
    if args.api_config is not None:
        api_config = Config(OpenAIAPI).load(args.api_config)
        api = ConfigurableFactory(OpenAIAPI).create(api_config)
        omit = {
            "summarizer": {"": {"api"}}
        }
        config = Config(OpenAISummarizerWorkflow, omit=omit).load(args.config)
        summarizer_workflow = ConfigurableFactory(OpenAISummarizerWorkflow, omit=omit).create(config)
        summarizer_workflow.args["summarizer"].args["api"] = api
        summarizer_workflow = summarizer_workflow.create()
    else:
        config = Config(OpenAISummarizerWorkflow).load(args.config)
        summarizer_workflow = ConfigurableFactory(OpenAISummarizerWorkflow).create(config)

    # get template fields from user
    template_fields = {}
    if isinstance(summarizer_workflow.summarizer, QueryBasedMultiDocSummarizer):
        print("Query-based multi-document summarizer.")
        # get query from user
        query = input("Enter query: ")

        # get documents from user
        docs = []
        while True:
            print("")
            doc = input("Enter document. Leave empty to finish: ")
            if doc == "":
                break
            docs.append(doc)

        # summarize
        template_fields["query"] = query
        template_fields["docs"] = docs

    elif isinstance(summarizer_workflow.summarizer, SingleDocSummarizer):
        print("Single document summarizer.")
        doc = input("Enter document: ")
        template_fields["doc"] = doc
    else:
        raise ValueError("Unsupported summarizer.")

    # summarize
    summarizer_workflow(**template_fields)


def create_summarizer_config(args):
    """
    Creates empty configuration for summarizer.

    :param args: Parsed arguments.
    """
    # make sure that all models are imported
    import semantsum.openai

    conv_subclasses = sorted(set(c.__name__ for c in subclasses(Summarizer)))
    summarizer = inquirer.prompt([
        inquirer.List('summarizer',
                      message="Select summarizer",
                      choices=conv_subclasses,
                      )
    ])["summarizer"]

    with open(args.path, "w") as f:
        config = Config(
            OpenAISummarizerWorkflow,
            file_override_user_defaults={
                "summarizer": {
                    "cls": sub_cls_from_its_name(Summarizer, summarizer),
                    "config": {}
                }
            })
        config.save(f)


def create_api_config(args):
    """
    Creates empty API configuration.

    :param args: Parsed arguments.
    """
    # make sure that all models are imported
    import semantsum.openai

    conv_subclasses = sorted(set(c.__name__ for c in subclasses(API)))
    api = inquirer.prompt([
        inquirer.List('api',
                      message="Select api",
                      choices=conv_subclasses,
                      )
    ])["api"]

    with open(args.path, "w") as f:
        config = Config(sub_cls_from_its_name(API, api))
        config.save(f)


def create_config(args):
    """
    Creates empty configurations.

    :param args: Parsed arguments.
    """
    config_type = inquirer.prompt([
        inquirer.List('config_type',
                      message="Which configuration do you want to create?",
                      choices=["summarization", "API"],
                      )
    ])["config_type"]

    if config_type == "summarization":
        create_summarizer_config(args)
    elif config_type == "API":
        create_api_config(args)
    else:
        raise ValueError("Unsupported configuration type.")


def main():
    parser = ArgumentParser(description="Summarizer for SemANT project.")
    subparsers = parser.add_subparsers()

    run_openai_summarizer_parser = subparsers.add_parser("run_openai_summarizer", help="Runs OpenAI summarizer.")
    run_openai_summarizer_parser.add_argument("config", help="Path to the configuration file.")
    run_openai_summarizer_parser.add_argument("--api_config", help="Path to the API configuration file. If not provided, API configuration will be read from the configuration file.", default=None)
    run_openai_summarizer_parser.set_defaults(func=run_openai_summarizer)

    create_config_parser = subparsers.add_parser("create_config", help="Creates configuration empty configuration file for specified summarizer.")
    create_config_parser.add_argument("path", help="Path to the configuration file.")
    create_config_parser.set_defaults(func=create_config)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
