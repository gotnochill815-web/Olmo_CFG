import torch

from src.generation.prompt_builder import build_prompt


def generate_multiple(
    model,
    tokenizer,
    qed,
    guidance_scale,
    n_samples=10,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=128,
    save_path=None,
):
    """
    Generate multiple molecules.

    Returns
    -------
    list[str]
        Generated SMILES.
    """

    molecules = []

    print("=" * 60)
    print(f"QED Target     : {qed}")
    print(f"Guidance Scale : {guidance_scale}")
    print(f"Samples        : {n_samples}")
    print("=" * 60)

    for i in range(n_samples):

        smiles = generate_molecule(
            model=model,
            tokenizer=tokenizer,
            qed=qed,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        molecules.append(smiles)

        print(f"[{i+1:02d}] {smiles}")

    if save_path is not None:
        with open(save_path, "w") as f:
            for smi in molecules:
                f.write(smi + "\n")

    return molecules