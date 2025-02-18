# Textjuicer - SemANTSum
This is a repository for summarization built as a part of the SemANT project. Textjuicer includes semantsum python package and REST API server both contained in this repository. Further, several language models finetuned for summarization are available from Huggingface. The models are compatible with Transformers library and Ollama.

## Installation
This can be installed as a standard python package:
    
```bash
pip install .
```

To use flash-attention, you need to also install the `flash-attn` package:

```bash
pip install flash-attn
```

## Configuration
The package is built in a way that makes every model fully loadable from a configuration file. The configuration file is a YAML file that contains all the necessary information to load the model.

You can use `create_config` argument to create a configuration file using the configuration builder:

```bash
./run.py create_config --path path_to_config.yaml
```

## How to add a new summarization model
For adding an API based summarization model the addition of a new model is as easy as creating a new configuration file and saving it in the `config` directory.

However, if you need to create a brand-new class for your model, read further.

The module `summarizer.py` contains abstract base classes for several types of summaries. If there is a new summarization model that does not fit into any of the existing classes, a new class can be added to the module. The new class should inherit from the `Summarizer` class.

To see an example of how to add a new model, you can inspect the `openai.py` module.

Also, check the `create_config` method in the `__main__.py` and `init_summarizer` in the `summarizator_factory.py` module to make your new model compatible with the configuration builder and loader.

## Demonstrator
There is a simple command line demonstrator showing how to use API based summarization models.

To run the OpenAI demonstrator, use the `run_summarizer` argument. You can use one of the configuration files from the `config` directory.

For these, you also need to configure the API. The API configuration could be provided in the model configuration or using `--api_config` parameter to specify a separate configuration file for the API.

```bash
./run.py run_summarizer semantsum/config/openai_query.yaml --api_config openai_api.yaml
```

To see an example of a structured (non-plaintext) summary, you can try:

```bash
./run.py run_summarizer semantsum/config/openai_query_timeline.yaml --api_config openai_api.yaml 
```


## List available models
There is an argument for listing available models:

```bash
./run.py available
``` 

An example output:
```
openai_query (openai)
openai_query_timeline (openai)
openai_query_topic (openai)
openai_single (openai)
ollama_query_tiny_llama (ollama)
```

## Overview of models on Hugging Face
List of models available on Hugging Face:

- `https://huggingface.co/BUT-FIT/csmpt-7B-RAGsum`
- `https://huggingface.co/BUT-FIT/CSTinyLLama-1.2B-RAGsum`

### Ollama notes
Models can be used with Ollama API after pulling them from Hugging Face. For example:

```bash
ollama pull hf.co/BUT-FIT/csmpt-7B-RAGsum
```

We tested our models with Ollama version **0.5.7**.