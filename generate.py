import argparse

from src.utils.config import load_yaml
from src.inference.load_model import load_model
from src.cfg.generator import generate_molecule

def main():

    parser = argparse.ArgumentParser(
        description="Conditional molecule generation using CFG"
    )

    parser.add_argument(
        "--model",
        default="configs/model/olmo_7b.yaml",
        help="Path to model config YAML",
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=2.0,
        help="Classifier-Free Guidance scale",
    )

    parser.add_argument(
        "--qed",
        type=float,
        default=None,
        help="Target QED",
    )

    parser.add_argument(
        "--logp",
        type=float,
        default=None,
        help="Target LogP",
    )

    parser.add_argument(
        "--tpsa",
        type=float,
        default=None,
        help="Target TPSA",
    )

    parser.add_argument(
        "--sas",
        type=float,
        default=None,
        help="Target SAS",
    )

    args = parser.parse_args()

    # Load model config
    model_cfg = load_yaml(args.model)

    # Load model and tokenizer
    model, tokenizer = load_model(model_cfg)

    # Generate molecule
    smiles = generate_molecule(
        model=model,
        tokenizer=tokenizer,
        guidance_scale=args.guidance,
        qed=args.qed,
        logp=args.logp,
        tpsa=args.tpsa,
        sas=args.sas,
    )

    print()
    print("=" * 80)
    print("Generated Molecule")
    print("=" * 80)
    print(smiles)


if __name__ == "__main__":
    main()