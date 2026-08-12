"""
CFG Decoder

Runs conditional and unconditional decoding using
Classifier-Free Guidance.

The unconditional prompt must match the training-time
CFG dropout representation.
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
    stop_newline: bool = True,
    verbose: bool = True,
):
    """
    Generate one molecule using CFG.
    """

    model.eval()

    device = next(model.parameters()).device

    if verbose:
        print("\n" + "=" * 80)
        print("Conditional Prompt")
        print(repr(conditional_prompt))

        print("\n" + "=" * 80)
        print("Unconditional Prompt")
        print(repr(unconditional_prompt))

        print("\n" + "=" * 80)
        print(f"CFG scale      : {guidance_scale}")
        print(f"Temperature    : {temperature}")
        print(f"Top-p          : {top_p}")
        print(f"Max new tokens : {max_new_tokens}")
        print(f"Stop newline   : {stop_newline}")
        print("=" * 80)

    cond_inputs = tokenizer(
        conditional_prompt,
        return_tensors="pt",
        clean_up_tokenization_spaces=False,
    ).to(device)

    uncond_inputs = tokenizer(
        unconditional_prompt,
        return_tensors="pt",
        clean_up_tokenization_spaces=False,
    ).to(device)

    cond_ids = cond_inputs["input_ids"]
    cond_mask = cond_inputs["attention_mask"]

    uncond_ids = uncond_inputs["input_ids"]
    uncond_mask = uncond_inputs["attention_mask"]

    prompt_len = cond_ids.shape[1]

    eos_id = tokenizer.eos_token_id

    newline_ids = set(
        tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
    )

    if verbose:
        print("\nConditional IDs:")
        print(cond_ids)

        print("\nConditional Tokens:")
        print(
            tokenizer.convert_ids_to_tokens(
                cond_ids[0]
            )
        )

        print("\nUnconditional IDs:")
        print(uncond_ids)

        print("\nUnconditional Tokens:")
        print(
            tokenizer.convert_ids_to_tokens(
                uncond_ids[0]
            )
        )

        print("\nEOS ID:", eos_id)
        print("Newline IDs:", sorted(newline_ids))

    for step in range(max_new_tokens):

        cond_outputs = model(
            input_ids=cond_ids,
            attention_mask=cond_mask,
        )

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

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        guided_logits = guided_logits / temperature

        probs = F.softmax(
            guided_logits,
            dim=-1,
        )

        if torch.isnan(probs).any():
            raise RuntimeError(
                "NaN detected in sampling probabilities."
            )

        # --------------------------------------------------------
        # TOP-P
        # --------------------------------------------------------

        if top_p < 1.0:

            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending=True,
                dim=-1,
            )

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            sorted_mask = cumulative_probs > top_p

            sorted_mask[..., 1:] = (
                sorted_mask[..., :-1].clone()
            )

            sorted_mask[..., 0] = False

            sorted_probs = sorted_probs.masked_fill(
                sorted_mask,
                0.0,
            )

            sorted_probs = (
                sorted_probs
                / sorted_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
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

        token_id = next_token.item()

        if verbose:
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            print(
                f"Step {step:03d} | "
                f"ID={token_id} | "
                f"Token={repr(token_text)}"
            )

        # --------------------------------------------------------
        # STOP ON EOS
        # --------------------------------------------------------

        if (
            eos_id is not None
            and token_id == eos_id
        ):
            if verbose:
                print("\nReached EOS.")
            break

        # --------------------------------------------------------
        # STOP ON NEWLINE
        # --------------------------------------------------------

        if (
            stop_newline
            and token_id in newline_ids
        ):
            if verbose:
                print(
                    "\nReached newline. "
                    "Stopping after first molecule."
                )
            break

        # --------------------------------------------------------
        # APPEND SAME GENERATED TOKEN TO BOTH BRANCHES
        # --------------------------------------------------------

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

    # ------------------------------------------------------------
    # DECODE GENERATED TOKENS
    # ------------------------------------------------------------

    generated_ids = cond_ids[0][prompt_len:]

    if verbose:
        print("\nGenerated IDs:")
        print(generated_ids.tolist())

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if verbose:
        print("\nRaw Decoded Output:")
        print(repr(raw_output))

    generated = raw_output

    if tokenizer.eos_token is not None:
        generated = generated.split(
            tokenizer.eos_token
        )[0]

    if stop_newline:
        generated = generated.split("\n")[0]

    generated = generated.strip()

    if verbose:
        print("\nFinal Output:")
        print(repr(generated))
        print("=" * 80)

    return generated
