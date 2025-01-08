from argparse import ArgumentParser

import inquirer
from classconfig import Config
from classconfig.classes import subclasses, sub_cls_from_its_name
from slugify import slugify

from semantsum import init_summarizer, list_summarizers
from semantsum.api import API
from semantsum.constants import CONFIG_DIR
from semantsum.openai import OpenAIAPI
from semantsum.summarizer import Summarizer, QueryBasedMultiDocSummarizer, SingleDocSummarizer, SummarizerWorkflow


def run_summarizer(args):
    """
    Runs summarizer.

    :param args: Parsed arguments.
    """

    # create summarizer from configuration
    if args.api_config is not None:
        api_config = Config(OpenAIAPI).load(args.api_config)
        summarizer = init_summarizer(args.config,
                                     api_key=api_config["api_key"],
                                     api_base_url=api_config.get("base_url", None))
    else:
        summarizer = init_summarizer(args.config)

    summary = None
    if isinstance(summarizer, QueryBasedMultiDocSummarizer):
        print("Query-based multi-document summarizer.")
        # get query from user
        query = input("Enter query: ")

        # get documents from user
        text = []
        while True:
            print("")
            t = input("Enter text. Leave empty to finish: ")
            if t == "":
                break
            text.append(t)

        summary = summarizer.summ_str(
            text=text,
            query=query
        )

    elif isinstance(summarizer, SingleDocSummarizer):
        print("Single document summarizer.")
        text = input("Enter text: ")
        summary = summarizer.summ_str([text])
    else:
        raise ValueError("Unsupported summarizer.")

    print("\nSummary:")
    print(summary)


def create_summarizer_config(args):
    """
    Creates empty configuration for summarizer.

    :param args: Parsed arguments.
    """
    # make sure that all models are imported
    import semantsum.openai # noqa
    import semantsum.local_hf # noqa

    conv_subclasses = sorted(set(c.__name__ for c in subclasses(Summarizer)))
    summarizer = inquirer.prompt([
        inquirer.List('summarizer',
                      message="Select summarizer",
                      choices=conv_subclasses,
                      )
    ])["summarizer"]
    summarizer_cls = sub_cls_from_its_name(Summarizer, summarizer)
    name = None
    while name is None or name in list_summarizers():
        if name is not None:
            print("Name already exists. Please choose another name.")
        name = inquirer.text(message="Enter configuration name")

    additional_args = {}
    if hasattr(summarizer_cls, "TYPE"):
        additional_args["default"] = summarizer_cls.TYPE
    summary_type = inquirer.text(message="Enter summary type", **additional_args)
    provider = inquirer.prompt([inquirer.List(
        "provider",
        message="Provider of the summarization",
        choices=["openai", "ollama", "local"]
    )])["provider"]
    description = inquirer.text(message="Enter configuration description")
    version = inquirer.text(message="Enter configuration version", default="1.0.0")
    recommended_num_texts = inquirer.text(message="Recommended number of texts for summarization", default=1,
                                            validate=lambda _, x: x.isdigit())
    recommended_num_texts = int(recommended_num_texts)

    if args.path is None:
        suggested_file_name = slugify(name, separator="_")
        file_name = inquirer.text(message="Enter file name without extension", default=suggested_file_name)
        args.path = CONFIG_DIR / f"{file_name}.yaml"

    with open(args.path, "w") as f:
        config = Config(
            SummarizerWorkflow,
            file_override_user_defaults={
                "name": name,
                "summary_type": summary_type,
                "provider": provider,
                "description": description,
                "version": version,
                "recommended_num_texts": recommended_num_texts,
                "summarizer": {
                    "cls": summarizer_cls,
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
        if args.path is None:
            raise ValueError("Path to the configuration file must be provided.")
        create_api_config(args)
    else:
        raise ValueError("Unsupported configuration type.")


def available(args):
    """
    Lists available configurations.

    :param args: Parsed arguments.
    """
    all_configs = list_summarizers()
    for name, config in all_configs.items():
        print(f"{name} ({config['provider']})")


def main():
    parser = ArgumentParser(description="Summarizer for SemANT project.")
    subparsers = parser.add_subparsers()

    run_summarizer_parser = subparsers.add_parser("run_summarizer", help="Runs summarizer.")
    run_summarizer_parser.add_argument("config", help="Path to the configuration file or name of the configuration.")
    run_summarizer_parser.add_argument("--api_config", help="Path to the API configuration file. If not provided, API configuration will be read from the configuration file.", default=None)
    run_summarizer_parser.set_defaults(func=run_summarizer)

    create_config_parser = subparsers.add_parser("create_config", help="Creates empty configuration file for specified summarizer.")
    create_config_parser.add_argument("--path", help="Path to the configuration file. If not provided it will be saved to the default configuration directory.", default=None)
    create_config_parser.set_defaults(func=create_config)

    create_config_parser = subparsers.add_parser("available", help="Lists available configurations.")
    create_config_parser.set_defaults(func=available)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
