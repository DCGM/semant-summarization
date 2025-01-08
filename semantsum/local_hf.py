from typing import Optional, Union, Sequence

from classconfig import ConfigurableValue, ConfigurableFactory, LoadedConfig, UsedConfig
from classconfig.validators import BoolValidator, FloatValidator, AnyValidator, ListOfTypesValidator, IsNoneValidator, \
    StringValidator

from semantsum.prompt_builder import PromptBuilder
from semantsum.summarizer import Summarizer, QueryBasedMultiDocSummarizer, SingleDocSummarizer
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFGenerationConfig(GenerationConfig):
    """
    Wrapper for Hugging Face generation configuration, so it can be loaded from the configuration file.

    Note: The description of the parameters is taken from the Hugging Face documentation and it is not guaranteed to be up-to-date.
    """
    used_config: LoadedConfig = UsedConfig()
    max_length: int = ConfigurableValue("The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.", user_default=20, voluntary=True)
    max_new_tokens: int = ConfigurableValue("The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.", user_default=None, voluntary=True)
    min_length: int = ConfigurableValue("The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.", user_default=0, voluntary=True)
    min_new_tokens: int = ConfigurableValue("The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.", user_default=None, voluntary=True)
    early_stopping: bool = ConfigurableValue("Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `\"never\"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).", user_default=False, voluntary=True)
    max_time: float = ConfigurableValue("The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed.", user_default=None, voluntary=True)
    stop_strings: str = ConfigurableValue("A string or a list of strings that should terminate generation if the model outputs them.", user_default=None, voluntary=True)
    do_sample: bool = ConfigurableValue("Whether or not to use sampling ; use greedy decoding otherwise.", user_default=False, voluntary=True)
    num_beams: int = ConfigurableValue("Number of beams for beam search. 1 means no beam search.", user_default=1, voluntary=True)
    num_beam_groups: int = ConfigurableValue("Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.", user_default=1, voluntary=True)
    penalty_alpha: float = ConfigurableValue("The values balance the model confidence and the degeneration penalty in contrastive search decoding.", user_default=None, voluntary=True)
    use_cache: bool = ConfigurableValue("Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.", user_default=True, voluntary=True)
    temperature: float = ConfigurableValue("The value used to modulate the next token probabilities.", user_default=1.0, voluntary=True)
    top_k: int = ConfigurableValue("The number of highest probability vocabulary tokens to keep for top-k-filtering.", user_default=50, voluntary=True)
    top_p: float = ConfigurableValue("If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.", user_default=1.0, voluntary=True)
    min_p: float = ConfigurableValue("Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in the 0.99-0.8 range (use the opposite of normal `top_p` values).", user_default=None, voluntary=True)
    typical_p: float = ConfigurableValue("Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation.", user_default=1.0, voluntary=True)
    epsilon_cutoff: float = ConfigurableValue("If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model.", user_default=0.0, voluntary=True)
    eta_cutoff: float = ConfigurableValue("Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.", user_default=0.0, voluntary=True)
    diversity_penalty: float = ConfigurableValue("This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.", user_default=0.0, voluntary=True)
    repetition_penalty: float = ConfigurableValue("The parameter for repetition penalty. 1.0 means no penalty.", user_default=1.0, voluntary=True)
    encoder_repetition_penalty: float = ConfigurableValue("The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.", user_default=1.0, voluntary=True)
    length_penalty: float = ConfigurableValue("Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.", user_default=1.0, voluntary=True)
    no_repeat_ngram_size: int = ConfigurableValue("If set to int > 0, all ngrams of that size can only occur once.", user_default=0, voluntary=True)
    bad_words_ids: list[list[int]] = ConfigurableValue("List of list of token ids that are not allowed to be generated.", user_default=None, voluntary=True)
    force_words_ids: list[list[int]] = ConfigurableValue("List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word.", user_default=None, voluntary=True)
    renormalize_logits: bool = ConfigurableValue("Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.", user_default=False, voluntary=True)
    forced_bos_token_id: int = ConfigurableValue("The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target language token.", user_default=None, voluntary=True)
    forced_eos_token_id: int = ConfigurableValue("The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens.", user_default=None, voluntary=True)
    remove_invalid_values: bool = ConfigurableValue("Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.", user_default=None, voluntary=True)
    exponential_decay_length_penalty: tuple = ConfigurableValue("This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay", user_default=None, voluntary=True)
    suppress_tokens: list[int] = ConfigurableValue("A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their log probs to `-inf` so that they are not sampled.", user_default=None, voluntary=True)
    begin_suppress_tokens: list[int] = ConfigurableValue("A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled.", user_default=None, voluntary=True)
    forced_decoder_ids: list[list[int]] = ConfigurableValue("A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token of index 123.", user_default=None, voluntary=True)
    sequence_bias: dict = ConfigurableValue("Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the sequence being selected, while negative biases do the opposite. Check [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.", user_default=None, voluntary=True)
    token_healing: bool = ConfigurableValue("Heal tail tokens of prompts by replacing them with their appropriate extensions. This enhances the quality of completions for prompts affected by greedy tokenization bias.", user_default=False, voluntary=True)
    guidance_scale: float = ConfigurableValue("The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality.", user_default=None, voluntary=True)
    low_memory: bool = ConfigurableValue("Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory. Used with beam search and contrastive search.", user_default=False, voluntary=True)
    watermarking_config: dict = ConfigurableValue("Arguments used to watermark the model outputs by adding a small bias to randomly selected set of 'green' tokens. If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally. See [this paper](https://arxiv.org/abs/2306.04634) for more details. Accepts the following keys: - greenlist_ratio (`float`): Used for watermarking. The ratio of 'green' tokens used to the vocabulary size. Defaults to 0.25. - bias (`float`): Used with watermarking. The bias added to the selected 'green' tokens' logits. Defaults to 2.0. - hashing_key (`int`): Hahsing key used for watermarking. Defaults to 15485863 (the millionth prime). - seeding_scheme (`str`): Algorithm to use for watermarking. Accepts values: - 'lefthash' (default): 'green' tokens selection depend on the last token (Algorithm 2 from the paper) - 'selfhash': 'green' tokens selection depends on the current token itself (Algorithm 3 from the paper) The downside of this scheme is that it considers all possible next tokens and can be slower than 'lefthash'. - context_width (`int`): The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.", user_default=None, voluntary=True)
    num_return_sequences: int = ConfigurableValue("The number of independently computed returned sequences for each element in the batch.", user_default=1, voluntary=True)
    output_attentions: bool = ConfigurableValue("Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more details.", user_default=False, voluntary=True)
    output_hidden_states: bool = ConfigurableValue("Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more details.", user_default=False, voluntary=True)
    output_scores: bool = ConfigurableValue("Whether or not to return the prediction scores. See `scores` under returned tensors for more details.", user_default=False, voluntary=True)
    output_logits: bool = ConfigurableValue("Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for more details.", user_default=None, voluntary=True)
    return_dict_in_generate: bool = ConfigurableValue("Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.", user_default=False, voluntary=True)
    pad_token_id: Union[int, list[int]] = ConfigurableValue("The id of the *padding* token.", user_default=None, voluntary=True)
    bos_token_id: Union[int, list[int]] = ConfigurableValue("The id of the *beginning-of-sequence* token.", user_default=None, voluntary=True)
    eos_token_id: Union[int, list[int]] = ConfigurableValue("The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.", user_default=None, voluntary=True)
    encoder_no_repeat_ngram_size: int = ConfigurableValue("If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.", user_default=0, voluntary=True)
    decoder_start_token_id: int = ConfigurableValue("If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length `batch_size`. Indicating a list enables different start ids for each element in the batch (e.g. multilingual models with different target languages in one batch)", user_default=None, voluntary=True)
    num_assistant_tokens: int = ConfigurableValue("Defines the number of _speculative tokens_ that shall be generated by the assistant model before being checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant model requires lots of corrections, lower speed-ups are reached.", user_default=5, voluntary=True)
    num_assistant_tokens_schedule: str = ConfigurableValue("Defines the schedule at which max assistant tokens shall be changed during inference. - \"heuristic\": When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model. - \"heuristic_transient\": Same as \"heuristic\" but `num_assistant_tokens` is reset to its initial value after each generation call. - \"constant\": `num_assistant_tokens` stays unchanged during generation", user_default="heuristic", voluntary=True)
    prompt_lookup_num_tokens: int = ConfigurableValue("The number of tokens to be output as candidate tokens.", user_default=None, voluntary=True)
    max_matching_ngram_size: int = ConfigurableValue("The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.", user_default=None, voluntary=True)
    dola_layers: Union[str, list[int]] = ConfigurableValue("The layers to use for DoLa decoding. If `None`, DoLa decoding is not used. If a string, it must be one of \"low\" or \"high\", which means using the lower part or higher part of the model layers, respectively. \"low\" means the first half of the layers up to the first 20 layers, and \"high\" means the last half of the layers up to the last 20 layers. If a list of integers, it must contain the indices of the layers to use for candidate premature layers in DoLa. The 0-th layer is the word embedding layer of the model. Set to 'low' to improve long-answer reasoning tasks, 'high' to improve short-answer tasks. Check the [documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/generation_strategies.md) or [the paper](https://arxiv.org/abs/2309.03883) for more details.", user_default=None, voluntary=True)
    cache_implementation: str = ConfigurableValue("Cache class that should be used when generating.", user_default=None, voluntary=True)
    cache_config: dict = ConfigurableValue("Arguments used in the key-value cache class can be passed in `cache_config`. Can be passed as a `Dict` and it will be converted to its repsective `CacheConfig` internally.", user_default=None, voluntary=True)
    return_legacy_cache: bool = ConfigurableValue("Whether to return the legacy or new format of the cache when `DynamicCache` is used by default.", user_default=True, voluntary=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConfigurableBitsAndBytesFactory:
    """
    Configuration factory for bits and bytes quantization.
    """

    load_in_8bit: bool = ConfigurableValue("This flag is used to enable 8-bit quantization with LLM.int8().",
                                           user_default=False,
                                           validator=BoolValidator(),
                                           voluntary=True
                                           )
    load_in_4bit: bool = ConfigurableValue("This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from `bitsandbytes`.",
                                           user_default=False,
                                           validator=BoolValidator(),
                                           voluntary=True
                                           )
    llm_int8_threshold: float = ConfigurableValue("This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).",
                                                  user_default=6.0,
                                                  validator=FloatValidator(),
                                                  voluntary=True
                                                  )
    llm_int8_skip_modules: Optional[list[str]] = ConfigurableValue("An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.",
                                                                   validator=AnyValidator([ListOfTypesValidator(str), IsNoneValidator()]),
                                                                   voluntary=True
                                                                   )
    llm_int8_enable_fp32_cpu_offload: bool = ConfigurableValue("This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8 operations will not be run on CPU.",
                                                                user_default=False,
                                                                validator=BoolValidator(),
                                                                voluntary=True
                                                               )
    llm_int8_has_fp16_weight: bool = ConfigurableValue("This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.",
                                                        user_default=False,
                                                       validator=BoolValidator(),
                                                       voluntary=True)
    bnb_4bit_compute_dtype: Optional[str] = ConfigurableValue("This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.",
                                                              user_default=None,
                                                              voluntary=True,
                                                              validator=AnyValidator([StringValidator(), IsNoneValidator()])
                                                              )
    bnb_4bit_quant_type: str = ConfigurableValue("This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`.",
                                                 user_default="fp4",
                                                 voluntary=True,
                                                 validator=StringValidator()
                                                 )
    bnb_4bit_use_double_quant: bool = ConfigurableValue("This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.",
                                                        user_default=False,
                                                        validator=BoolValidator(),
                                                        voluntary=True
                                                        )
    bnb_4bit_quant_storage: Optional[str] = ConfigurableValue("This sets the storage type to pack the quanitzed 4-bit prarams.",
                                                                user_default=None,
                                                                voluntary=True,
                                                                validator=AnyValidator([StringValidator(), IsNoneValidator()])
                                                                )

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.kwargs)


class HFWithPromptBuilder(Summarizer):
    """
    Base class for Hugging Face summarizers that use prompt builder to allow user to define configurable prompt.
    """

    model: str = ConfigurableValue("Name of model that should be used.", user_default="BUT-FIT/csmpt-7B-RAGsum")
    prompt_builder: PromptBuilder = ConfigurableFactory(PromptBuilder, "Prompt builder. Available fields: text, query. Text is list of strings and query is optional string (None when missing).")
    generation_config: Optional[HFGenerationConfig] = ConfigurableFactory(
        HFGenerationConfig,
        "Generation configuration for the model, it overrides the default generation configuration.",
        voluntary=True
    )
    cache_dir: Optional[str] = ConfigurableValue("Path to Hugging Face cache directory.", user_default=None, voluntary=True)
    quantization: Optional[ConfigurableBitsAndBytesFactory] = ConfigurableFactory(ConfigurableBitsAndBytesFactory,
                                                                                  "Configuration for bits and bytes quantization.",
                                                                                  voluntary=True)
    torch_dtype: Optional[str] = ConfigurableValue(
        "Override the default torch.dtype and load the model under a specific dtype",
        user_default=None,
        voluntary=True)
    attn_implementation: Optional[str] = ConfigurableValue(
        'The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual "eager" implementation.',
        user_default=None,
        voluntary=True)
    device_map: str = ConfigurableValue("Device map for model loading.", user_default="auto", voluntary=True)

    def __init__(self, model: str, prompt_builder: PromptBuilder, generation_config: Optional[HFGenerationConfig] = None, cache_dir: Optional[str] = None,
                 quantization: Optional[ConfigurableBitsAndBytesFactory] = None, torch_dtype: Optional[str] = None,
                 attn_implementation: Optional[str] = None, device_map: str = "auto"):
        self.model_path = model
        self.prompt_builder = prompt_builder
        self.generation_config = generation_config
        self.cache_dir = cache_dir
        self.quantization = quantization
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map

        additional_args_for_model = {}
        if isinstance(self.quantization, ConfigurableBitsAndBytesFactory):
            additional_args_for_model["quantization_config"] = self.quantization.create()

        if self.torch_dtype is not None:
            additional_args_for_model["torch_dtype"] = self.torch_dtype

        if self.attn_implementation is not None:
            additional_args_for_model["attn_implementation"] = self.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            **additional_args_for_model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir)

        self.generation_config = HFGenerationConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)

        if generation_config is not None:
            self.generation_config.update(**generation_config.used_config.untransformed)

    def summ_str(self, text: Sequence[str], query: Optional[str] = None) -> str:
        prompt = self.prompt_builder.build({"text": text, "query": query})

        model_input = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        output = self.model.generate(
            inputs=model_input["input_ids"].to(self.model.device),
            generation_config=self.generation_config,
            return_dict_in_generate=True
        )
        # get just the output
        output = output.sequences[0][model_input["input_ids"].shape[1]:]

        return self.tokenizer.decode(output, skip_special_tokens=True)


class HFSingleDocSummarizer(SingleDocSummarizer, HFWithPromptBuilder):
    ...


class HFQueryBasedMultiDocSummarizer(QueryBasedMultiDocSummarizer, HFWithPromptBuilder):
    ...
