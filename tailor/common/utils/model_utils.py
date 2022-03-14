import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from tailor.common.utils.head_prompt_utils import BadGenerationError, fillin_prompt


def load_generator(model_path="allenai/tailor"):
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    # TODO: use cached_path
    return pipeline(
        "text2text-generation",
        model=AutoModelForSeq2SeqLM.from_pretrained(model_path),
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        framework="pt",
        device=cuda_device,
    )


def generate_batch(
    examples,
    generator,
    temperature=0.75,
    top_k=50,
    num_beams=None,
    n=3,
    top_p=0.9,
    do_sample=True,
    batch_size=128,
    max_length=200,
    **kwargs
):
    preds_list = []
    with torch.no_grad():
        for e in range(0, len(examples), batch_size):
            preds_list += generator(
                examples[e: e + batch_size],
                temperature=temperature,
                return_tensors=True,
                num_beams=num_beams,
                top_p=top_p,
                top_k=top_k,
                max_length=max_length,
                early_stopping=None if num_beams is None else True,
                do_sample=num_beams is None and do_sample,
                num_return_sequences=n,
                **kwargs
            )
    return preds_list


def generate_and_clean_batch(
    prompts,
    generator,
    temperature=0.75,
    n=3,
    do_sample=True,
    top_p=0.9,
    top_k=50,
    batch_size=128,
    num_beams=None,
    is_clean_verb_prefix=True,
    max_length=200,
    **kwargs
):
    preds_list = generate_batch(
        prompts,
        generator,
        temperature=temperature,
        n=n,
        num_beams=num_beams,
        top_p=top_p,
        no_repeat_ngram_size=2,
        do_sample=do_sample,
        batch_size=batch_size,
        top_k=top_k,
        max_length=max_length,
        **kwargs
    )
    preds_list_cleaned = []
    for idx in range(0, len(preds_list), n):
        results = []
        prompt = prompts[int(idx / n)]
        for g in preds_list[idx: idx + n]:
            try:
                generated = generator.tokenizer.decode(g["generated_token_ids"])
                filled = fillin_prompt(prompt, generated, is_clean_prefix=is_clean_verb_prefix)
                results.append(filled)
            except BadGenerationError as e:

                # results.append([])
                continue
        preds_list_cleaned.append(list(set(results)))
    assert len(preds_list_cleaned) == len(prompts)
    return preds_list_cleaned
