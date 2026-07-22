"""
Generator for Classifier-Free Guidance (CFG).
"""

from src.cfg.decoder import cfg_decode
from src.cfg.prompt_builder import build_prompt


def generate_molecule(
    model,
    tokenizer,
    qed=None,
    logp=None,
    tpsa=None,
    sas=None,
    guidance_scale=2.0,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
):
    """
    Generate a single molecule using Classifier-Free Guidance (CFG).
    """

    # Conditional prompt
    conditional_prompt = build_prompt(
        qed=qed,
        logp=logp,
        tpsa=tpsa,
        sas=sas,
    )

    # Unconditional prompt
    unconditional_prompt = build_prompt()

    smiles = cfg_decode(
        model=model,
        tokenizer=tokenizer,
        conditional_prompt=conditional_prompt,
        unconditional_prompt=unconditional_prompt,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    return smiles


def generate_multiple(
    model,
    tokenizer,
    n_samples=10,
    qed=None,
    logp=None,
    tpsa=None,
    sas=None,
    guidance_scale=2.0,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    verbose=False,
):
    """
    Generate multiple molecules using CFG.
    """

    molecules = []

    print("=" * 60)
    print(f"Samples         : {n_samples}")
    print(f"Guidance Scale  : {guidance_scale}")
    print(f"QED             : {qed}")
    print(f"LOGP            : {logp}")
    print(f"TPSA            : {tpsa}")
    print(f"SAS             : {sas}")
    print("=" * 60)

    for i in range(n_samples):

        smiles = generate_molecule(
            model=model,
            tokenizer=tokenizer,
            qed=qed,
            logp=logp,
            tpsa=tpsa,
            sas=sas,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        molecules.append(smiles)

        if verbose:
            print(f"[{i+1:03d}] {smiles}")

    return molecules