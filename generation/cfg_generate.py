
import os
import csv
import json
from typing import Dict, List, Optional

from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt


PROPERTY_ORDER = [
    "qed",
    "logp",
    "tpsa",
    "sas",
]


def build_cfg_prompts(
    properties: Dict[str, Optional[float]]
):
    """
    Build conditional and unconditional prompts.

    Conditional:
        <pstart>
        <QED> ...
        <LOGP> ...
        <molstart>

    Unconditional:
        <pstart>
        <molstart>
    """

    conditional_prompt = build_prompt(
        qed=properties.get("qed"),
        logp=properties.get("logp"),
        tpsa=properties.get("tpsa"),
        sas=properties.get("sas"),
    )

    unconditional_prompt = build_prompt()

    return conditional_prompt, unconditional_prompt


def generate_one(
    model,
    tokenizer,
    properties: Dict[str, Optional[float]],
    guidance_scale: float,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
    verbose: bool = False,
):
    """
    Generate one molecule using classifier-free guidance.
    """

    conditional_prompt, unconditional_prompt = build_cfg_prompts(
        properties
    )

    smiles = cfg_decode(
        model=model,
        tokenizer=tokenizer,
        conditional_prompt=conditional_prompt,
        unconditional_prompt=unconditional_prompt,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_newline=True,
        verbose=verbose,
    )

    return smiles


def generate_condition(
    model,
    tokenizer,
    properties: Dict[str, Optional[float]],
    guidance_scale: float,
    n_samples: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
):
    """
    Generate n_samples molecules for one condition.
    """

    rows = []

    for sample_idx in range(n_samples):

        try:

            smiles = generate_one(
                model=model,
                tokenizer=tokenizer,
                properties=properties,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                verbose=False,
            )

        except Exception as e:

            print(
                f"[WARNING] Generation failed "
                f"sample={sample_idx}: {e}"
            )

            smiles = ""

        row = {
            "smiles": smiles,
            "guidance_scale": guidance_scale,
        }

        for property_name in PROPERTY_ORDER:

            value = properties.get(property_name)

            if value is not None:
                row[f"{property_name}_condition"] = value

        rows.append(row)

    return rows


def save_rows(
    rows: List[Dict],
    output_path: str,
):
    """
    Save generation results as CSV.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} generations → "
        f"{output_path}"
    )
