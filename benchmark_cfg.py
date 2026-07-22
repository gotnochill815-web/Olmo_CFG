import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

from src.cfg.generator import generate_multiple


def compute_properties(smiles_list, guidance):

    rows = []

    for i, smi in enumerate(smiles_list, start=1):

        mol = Chem.MolFromSmiles(smi)

        if mol is None:

            rows.append({
                "Guidance": guidance,
                "Molecule": i,
                "SMILES": smi,
                "Valid": False,
                "QED": None,
                "LogP": None,
                "TPSA": None,
            })

            continue

        rows.append({
            "Guidance": guidance,
            "Molecule": i,
            "SMILES": smi,
            "Valid": True,
            "QED": QED.qed(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
        })

    return pd.DataFrame(rows)


def benchmark(
    model,
    tokenizer,
    qed=0.90,
    n_samples=10,
):

    all_results = []

    guidance_scales = [
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
    ]

    for scale in guidance_scales:

        print("\n")
        print("=" * 80)
        print(f"Running CFG = {scale}")
        print("=" * 80)

        molecules = generate_multiple(
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            qed=qed,
            guidance_scale=scale,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=128,
            verbose=False,
)

        df = compute_properties(
            molecules,
            scale,
        )

        all_results.append(df)

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results.to_csv(
        "cfg_results.csv",
        index=False,
    )

    summary = (
        results
        .groupby("Guidance")
        .agg(
            Total=("Valid", "size"),
            Valid=("Valid", "sum"),
            Mean_QED=("QED", "mean"),
            Std_QED=("QED", "std"),
            Mean_LogP=("LogP", "mean"),
            Mean_TPSA=("TPSA", "mean"),
        )
    )

    summary["Validity (%)"] = (
        summary["Valid"] /
        summary["Total"] * 100
    )

    print("\n")
    print(summary)

    summary.to_csv(
        "cfg_summary.csv"
    )

    return results, summary