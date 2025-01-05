from typing import List

from . import summary_route
from summarization_api.schemas import base_objects
from summarization_api.config import config
from fastapi import HTTPException

import semantsum


@summary_route.get("/summarizers", response_model=List[base_objects.SummarizationConfig], tags=["Summary"])
async def get_summarizers():
    configs = semantsum.list_summarizers()
    for key, value in configs.items():
        value["name"] = key
    configs = [base_objects.SummarizationConfig.model_validate(config) for config in configs.values()]
    return configs


def get_url_and_key(provider: str) -> tuple[str, str]:
    if provider == "openai":
        return config.OPENAI_URL, config.OPENAI_KEY
    elif provider == "ollama":
        return config.OLLAMA_URL, config.OLLAMA_KEY
    else:
        raise HTTPException(status_code=404, detail=f"Provider {provider} not found.")


def init_summarizer(summarizer_name: str):
    configs = semantsum.list_summarizers()
    if summarizer_name not in configs:
        raise HTTPException(status_code=404, detail=f"Summarizer {summarizer_name} not found")

    configs = configs[summarizer_name]

    url, key = get_url_and_key(configs["provider"])
    summarizer = semantsum.init_summarizer(
        summarizer_name,
        api_key=key,
        api_base_url=url)
    return summarizer, configs


@summary_route.post("/string/{summarizer_name}", response_model=base_objects.SummarizationStringResponse, tags=["Summary"])
async def summarize_text(summarizer_name: str, text: List[str], query: str = None):
    summarizer, info = init_summarizer(summarizer_name)
    messages = []

    if 'query-based' not in info['summary_type'] and query is not None:
        query = None
        messages.append(f"Query is ignored as it is not supported by the summarizer {summarizer_name}.")

    if 'multi-document' not in info['summary_type'] and len(text) > 1:
        text = ['\n'.join(text)]
        messages.append(f"Texts are concatenated as the summarizer {summarizer_name} does not natively support multi-document summarization.")

    summary = summarizer.summ_str(text, query)

    messages = '\n'.join(messages)
    status = "OK" if not messages else "WARNING"
    summary = base_objects.SummarizationStringResponse(summary=summary, info=messages, status=status)
    return summary


@summary_route.post("/struct/{summarizer_name}", response_model=base_objects.SummarizationStructResponse, tags=["Summary"])
async def summarize_text(summarizer_name: str, text: List[str], query: str = None):
    summarizer, info = init_summarizer(summarizer_name)
    messages = []

    if 'query-based' not in info['summary_type'] and query is not None:
        query = None
        messages.append(f"Query is ignored as it is not supported by the summarizer {summarizer_name}.")

    if 'multi-document' not in info['summary_type'] and len(text) > 1:
        text = ['\n'.join(text)]
        messages.append(f"Texts are concatenated as the summarizer {summarizer_name} does not natively support multi-document summarization.")

    summary = summarizer.summ_struct(text, query)
    messages = '\n'.join(messages)
    status = "OK" if not messages else "WARNING"
    summary = [base_objects.SummaryItem(item=item, text=text) for item, text in summary]
    summary = base_objects.SummarizationStructResponse(summary=summary, info=messages, status=status)
    return summary
