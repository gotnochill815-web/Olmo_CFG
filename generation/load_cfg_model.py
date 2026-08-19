
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# MODELS
# ============================================================

BASE_MODEL = (
    "harindhar10/"
    "OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos"
)

ADAPTER_MODEL = (
    "prakhya15/"
    "Olmo-CFG-50000"
)

ADAPTER_SUBFOLDER = "adapter"


# ============================================================
# SPECIAL TOKENS
# ============================================================

SPECIAL_TOKENS = [
    "<QED>",
    "<LOGP>",
    "<TPSA>",
    "<SAS>",
    "<NULL>",
    "<pstart>",
    "<molstart>",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_cfg_model():

    print("=" * 80)
    print("LOADING OLMo-CFG-50000")
    print("=" * 80)

    print("\nBase model:")
    print(BASE_MODEL)

    print("\nCFG adapter:")
    print(
        f"{ADAPTER_MODEL}/{ADAPTER_SUBFOLDER}"
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is not available. "
            "Please enable a GPU runtime in Colab."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    print(
        "Original tokenizer size:",
        len(tokenizer)
    )

    # Add exactly the tokens used by CFG training
    num_added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens":
                SPECIAL_TOKENS
        }
    )

    print(
        "Special tokens added:",
        num_added
    )

    print(
        "Final tokenizer size:",
        len(tokenizer)
    )

    # --------------------------------------------------------
    # VERIFY SPECIAL TOKENS
    # --------------------------------------------------------

    print("\nSpecial token IDs:")

    for token in SPECIAL_TOKENS:

        token_id = tokenizer.convert_tokens_to_ids(
            token
        )

        print(
            f"{token:12s} -> {token_id}"
        )

    # --------------------------------------------------------
    # PAD TOKEN
    # --------------------------------------------------------

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # 4-BIT QUANTIZATION
    # --------------------------------------------------------

    print("\nConfiguring 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    print("\nLoading base OLMo-7B...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # IMPORTANT:
    # tokenizer has additional CFG tokens
    model.resize_token_embeddings(
        len(tokenizer)
    )

    print(
        "Base model loaded."
    )

    # --------------------------------------------------------
    # LOAD LORA ADAPTER
    # --------------------------------------------------------

    print("\nLoading CFG LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_MODEL,
        subfolder=ADAPTER_SUBFOLDER,
        is_trainable=False,
    )

    model.eval()

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print("\nAdapter loaded successfully.")

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    print(
        "Total parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print(
        "\nModel device:",
        next(model.parameters()).device
    )

    print(
        "Model dtype:",
        next(model.parameters()).dtype
    )

    print("\n" + "=" * 80)
    print("OLMo-CFG MODEL READY")
    print("=" * 80)

    return model, tokenizer
