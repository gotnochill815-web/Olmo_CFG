
"""
Final QED-conditioned CFG generation.

Final benchmark configuration:
    target property : QED
    min_new_tokens   : 8
    stop separators  : True
    block repeats    : False
    balance parens   : False
"""

from typing import Dict, Optional

from generation.cfg_generate import build_cfg_prompts
from src.cfg.decoder import cfg_decode


def generate_qed_molecule(
    model,
    tokenizer,
    target_qed: float = 0.50,
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

    properties: Dict[str, Optional[float]] = {
        "qed": target_qed,
        "logp": None,
        "tpsa": None,
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
