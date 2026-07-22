"""
CFG Decoder

Runs conditional and unconditional decoding using
Classifier-Free Guidance.
"""

import torch
import torch.nn.functional as F

from src.cfg.guidance import classifier_free_guidance


@torch.no_grad()
def cfg_decode(
    model,
    tokenizer,
    conditional_prompt: str,
    unconditional_prompt: str,
    guidance_scale: float = 2.0,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
):
    """
    Generate a molecule using Classifier-Free Guidance (CFG).
    """

    model.eval()

    device = next(model.parameters()).device

    #########################################################
    # DEBUG
    #########################################################

    print("\n" + "=" * 80)
    print("Conditional Prompt")
    print(repr(conditional_prompt))

    print("\n" + "=" * 80)
    print("Unconditional Prompt")
    print(repr(unconditional_prompt))
    print("=" * 80)

    #########################################################

    cond_inputs = tokenizer(
        conditional_prompt,
        return_tensors="pt",
    ).to(device)

    uncond_inputs = tokenizer(
        unconditional_prompt,
        return_tensors="pt",
    ).to(device)

    cond_ids = cond_inputs["input_ids"]
    cond_mask = cond_inputs["attention_mask"]

    uncond_ids = uncond_inputs["input_ids"]
    uncond_mask = uncond_inputs["attention_mask"]

    #########################################################
    # DEBUG
    #########################################################

    print("\nConditional IDs:")
    print(cond_ids)

    print("\nConditional Tokens:")
    print(tokenizer.convert_ids_to_tokens(cond_ids[0]))

    print("\nUnconditional IDs:")
    print(uncond_ids)

    print("\nUnconditional Tokens:")
    print(tokenizer.convert_ids_to_tokens(uncond_ids[0]))

    #########################################################

    prompt_len = cond_ids.shape[1]

    eos_id = tokenizer.eos_token_id

    print("\nEOS ID:", eos_id)

    for step in range(max_new_tokens):

        #########################################################
        # Conditional
        #########################################################

        cond_outputs = model(
            input_ids=cond_ids,
            attention_mask=cond_mask,
        )

        #########################################################
        # Unconditional
        #########################################################

        uncond_outputs = model(
            input_ids=uncond_ids,
            attention_mask=uncond_mask,
        )

        cond_logits = cond_outputs.logits[:, -1, :]
        uncond_logits = uncond_outputs.logits[:, -1, :]

        guided_logits = classifier_free_guidance(
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            guidance_scale=guidance_scale,
        )

        if temperature != 1.0:
            guided_logits = guided_logits / temperature

        probs = F.softmax(guided_logits, dim=-1)

        #########################################################
        # Top-p sampling
        #########################################################

        if top_p < 1.0:

            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending=True,
            )

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            sorted_probs[sorted_mask] = 0.0

            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1,
                keepdim=True,
            )

            sampled = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )

            next_token = sorted_indices.gather(
                -1,
                sampled,
            )

        else:

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        #########################################################
        # DEBUG
        #########################################################

        token_text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=False,
        )

        print(
            f"Step {step:03d} | "
            f"ID={next_token.item()} | "
            f"Token={repr(token_text)}"
        )

        #########################################################

        cond_ids = torch.cat(
            [cond_ids, next_token],
            dim=1,
        )

        uncond_ids = torch.cat(
            [uncond_ids, next_token],
            dim=1,
        )

        cond_mask = torch.cat(
            [
                cond_mask,
                torch.ones(
                    (cond_mask.size(0), 1),
                    dtype=cond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        uncond_mask = torch.cat(
            [
                uncond_mask,
                torch.ones(
                    (uncond_mask.size(0), 1),
                    dtype=uncond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        if next_token.item() == eos_id:
            print("\nReached EOS.")
            break

    #########################################################
    # Decode
    #########################################################

    generated_ids = cond_ids[0][prompt_len:]

    print("\nGenerated IDs:")
    print(generated_ids.tolist())

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    print("\nRaw Decoded Output:")
    print(repr(raw_output))

    generated = raw_output

    if tokenizer.eos_token is not None:
        generated = generated.split(tokenizer.eos_token)[0]

    generated = generated.strip()

    print("\nFinal Output:")
    print(repr(generated))
    print("=" * 80)

    return generated