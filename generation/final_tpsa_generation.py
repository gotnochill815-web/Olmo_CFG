
"""
Final TPSA-conditioned CFG generation.

Clean benchmark configuration:
    - min_new_tokens = 8
    - stop_separators = True
    - block_repeats = False
    - balance_parens = False
"""

from src.cfg.decoder import cfg_decode
from generation.cfg_generate import build_cfg_prompts


def generate_tpsa_molecule(
    model,
    tokenizer,
    target_tpsa: float = 50.0,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
    min_new_tokens: int = 8,
    stop_separators: bool = True,
    block_repeats: bool = False,
    repeat_ngram_size: int = 4,
    balance_parens: bool = False,
    verbose: bool = False,
) -> str:

    properties = {
        "qed": None,
        "logp": None,
        "tpsa": target_tpsa,
        "sas": None,
    }

    conditional_prompt, unconditional_prompt = (
        build_cfg_prompts(properties)
    )

    return cfg_decode(
        model=model,
        tokenizer=tokenizer,
        conditional_prompt=conditional_prompt,
        unconditional_prompt=unconditional_prompt,
        guidance_scale=guidance_scale,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_newline=True,
        stop_separators=stop_separators,
        block_repeats=block_repeats,
        repeat_ngram_size=repeat_ngram_size,
        balance_parens=balance_parens,
        verbose=verbose,
    )
