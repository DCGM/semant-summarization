from os import PathLike
from pathlib import Path
from typing import Optional

from classconfig import Config, ConfigurableFactory, YAML

from semantsum.constants import CONFIG_DIR
from semantsum.summarizer import Summarizer, SummarizerWorkflow


def list_summarizers() -> dict[str, dict[str, str]]:
    """
    Gets all types of summarizers.

    :return: dictionary with following keys:
        - name
            - summary_type
            - provider
            - description
            - version
            - recommended_num_texts
            - config_path
    :raises ValueError: if there are two summarizers with the same name
    """

    all_configs = {}
    for config_path in CONFIG_DIR.glob("*.yaml"):
        with open(config_path, "r") as f:
            config = YAML().load(f)

        if config["name"] in all_configs:
            raise ValueError(f"There are two configurations with the same name: {config['name']}\n{config_path}\n{all_configs[config['name']]['config_path']}")

        filename = config_path.stem
        all_configs[filename] = {
            "summary_type": config["summary_type"],
            "provider": config["provider"],
            "description": config["description"],
            "version": config["version"],
            "recommended_num_texts": config["recommended_num_texts"],
            "config_path": str(config_path)
        }

    return all_configs


def init_summarizer(config: str | PathLike | dict, api_key: Optional[str] = None, api_base_url: Optional[str] = None) -> Summarizer:
    """
    Initializes summarizer from configuration.

    :param config:
        Name of configuration
        Path to configuration file
        Configuration dictionary
    :param api_key: API key.
        Setting this parameter has higher priority than setting it in configuration.
        The summarizer must be API-based and have an API attribute.
    :param api_base_url: API base URL.
        Setting this parameter has higher priority than setting it in configuration.
        The summarizer must be API-based and have an API attribute.
    :return: Summarizer.
    :raises ValueError: if configuration is not found

    """

    # make sure that all models are imported
    import semantsum.openai  # noqa
    import semantsum.local_hf  # noqa

    path_to_config = None
    # load YAML
    if not isinstance(config, dict):
        check = Path(config)

        if check.is_file():
            path_to_config = check
        else:
            available_configs = list_summarizers()

            if config not in available_configs:
                raise ValueError(f"Configuration {config} not found.")

            path_to_config = available_configs[config]["config_path"]

        with open(path_to_config, "r") as f:
            config = YAML().load(f)

    # update config with additional parameters

    if api_key is not None or api_base_url is not None:
        if "api" not in config["summarizer"]["config"]:
            raise ValueError("The summarizer is not API-based or the API attribute is not present.")

        if api_key is not None:
            config["summarizer"]["config"]["api"]["api_key"] = api_key
        if api_base_url is not None:
            config["summarizer"]["config"]["api"]["base_url"] = api_base_url

    # validate and transform config
    config = Config(SummarizerWorkflow, allow_extra=False).trans_and_val(config, path_to=path_to_config)

    # create summarizer
    return ConfigurableFactory(SummarizerWorkflow).create(config).summarizer
