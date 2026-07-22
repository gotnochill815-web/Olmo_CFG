"""
Model loading utilities.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def load_model(model_cfg):
    """
    Load tokenizer and LoRA-ready model.
    """

    model_name = model_cfg["model_name"]

    # -----------------------
    # Tokenizer
    # -----------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_tokens = model_cfg["special_tokens"]

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": special_tokens
        }
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------
    # Quantization
    # -----------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # -----------------------
    # Base model
    # -----------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=model_cfg["device_map"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    model.resize_token_embeddings(len(tokenizer))

    # -----------------------
    # Gradient checkpointing
    # -----------------------

    if model_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    # -----------------------
    # LoRA
    # -----------------------

    lora_cfg = LoraConfig(
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    return model, tokenizer