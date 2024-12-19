# SemANTSum
This is a repository for summarization built as a part of the SemANT project.

## Configuration
The package is built in a way that makes every model fully loadable from a configuration file. The configuration file is a YAML file that contains all the necessary information to load the model.

You can use `create_config` argument to create a configuration file using the configuration builder:

```bash
./run.sh create_config path_to_config.yaml
```

## How to add a new summarization model

The module `summarizer.py` contains abstract base classes for several types of summaries. If there is a new summarization model that does not fit into any of the existing classes, a new class can be added to the module. The new class should inherit from the `Summarizer` class.

To see an example of how to add a new model, you can inspect the `openai.py` module.

Also, check the `create_config` method in the `__main__.py` module to make your new model compatible with the configuration builder.

## Demonstrators
There are several demonstrators showing how to use given models.

### OpenAI
To run the OpenAI demonstrator, use the `run_openai_summarizer` argument. You can use one of the configuration files from the `config` directory.

For these, you also need to configure the API. The API configuration could be provided in the model configuration or using `--api_config` parameter to specify separate configuration file for the API.

```bash
./run.py run_openai_summarizer semantsum/config/openai_query_based_multi_doc.yaml --api_config openai_api.yaml
```

To see an example of a structured (non-plaintext) summary, you can try:

```bash
./run.py run_openai_summarizer semantsum/config/openai_query_based_multi_doc_timeline.yaml --api_config openai_api.yaml 
```