"""
CFG Decoder

Runs conditional and unconditional decoding using
Classifier-Free Guidance.

The unconditional prompt must match the training-time
CFG dropout representation.

Termination safeguards:
    - EOS stopping
    - newline stopping
    - ';' stopping
    - ',' stopping
    - minimum generated-token protection
    - defensive post-decoding separator cleanup
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
    min_new_tokens: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    stop_newline: bool = True,
    stop_separators: bool = True,
    verbose: bool = True,
):
    """
    Generate one molecule using Classifier-Free Guidance.

    Parameters
    ----------
    model:
        Loaded causal language model.

    tokenizer:
        Tokenizer corresponding to the model.

    conditional_prompt:
        Prompt containing the requested molecular properties.

    unconditional_prompt:
        Prompt representing the CFG dropout/unconditional condition.

    guidance_scale:
        CFG guidance strength.

    max_new_tokens:
        Maximum number of molecular tokens to generate.

    min_new_tokens:
        Minimum number of molecular tokens before EOS/newline/
        separator termination is allowed.

    temperature:
        Sampling temperature.

    top_p:
        Nucleus sampling threshold.

    stop_newline:
        Stop when newline is generated.

    stop_separators:
        Stop when ';' or ',' is generated.

    verbose:
        Print detailed generation diagnostics.
    """

    model.eval()

    device = next(model.parameters()).device

    # ============================================================
    # CONFIGURATION
    # ============================================================

    if guidance_scale < 0:
        raise ValueError(
            "guidance_scale must be >= 0."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0."
        )

    if not 0 <= top_p <= 1:
        raise ValueError(
            "top_p must be in [0, 1]."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be > 0."
        )

    if min_new_tokens < 0:
        raise ValueError(
            "min_new_tokens must be >= 0."
        )

    if min_new_tokens > max_new_tokens:
        raise ValueError(
            "min_new_tokens cannot be greater than "
            "max_new_tokens."
        )

    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================

    if verbose:

        print("\n" + "=" * 80)
        print("Conditional Prompt")
        print(repr(conditional_prompt))

        print("\n" + "=" * 80)
        print("Unconditional Prompt")
        print(repr(unconditional_prompt))

        print("\n" + "=" * 80)
        print(f"CFG scale       : {guidance_scale}")
        print(f"Temperature     : {temperature}")
        print(f"Top-p           : {top_p}")
        print(f"Max new tokens  : {max_new_tokens}")
        print(f"Min new tokens  : {min_new_tokens}")
        print(f"Stop newline    : {stop_newline}")
        print(f"Stop separators : {stop_separators}")
        print("=" * 80)

    # ============================================================
    # TOKENIZE CONDITIONAL PROMPT
    # ============================================================

    cond_inputs = tokenizer(
        conditional_prompt,
        return_tensors="pt",
    ).to(device)

    # ============================================================
    # TOKENIZE UNCONDITIONAL PROMPT
    # ============================================================

    uncond_inputs = tokenizer(
        unconditional_prompt,
        return_tensors="pt",
    ).to(device)

    cond_ids = cond_inputs["input_ids"]
    cond_mask = cond_inputs["attention_mask"]

    uncond_ids = uncond_inputs["input_ids"]
    uncond_mask = uncond_inputs["attention_mask"]

    # Number of tokens belonging to the conditional prompt.
    # Everything generated after this point is the molecule.
    prompt_len = cond_ids.shape[1]

    # ============================================================
    # SPECIAL TOKEN IDS
    # ============================================================

    eos_id = tokenizer.eos_token_id

    newline_ids = set(
        tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
    )

    # ============================================================
    # STOP IDS
    # ============================================================

    # Newline is always a termination signal when enabled.
    stop_ids = set()

    if stop_newline:
        stop_ids.update(
            newline_ids
        )

    # Treat semicolon/comma as molecule separators.
    #
    # This prevents accidental generation of:
    #
    #     molecule1;molecule2
    #
    # or
    #
    #     molecule1,molecule2
    #
    # which was one of the invalid-generation patterns
    # observed during the CFG experiments.

    if stop_separators:

        for stop_char in [";", ","]:

            ids = tokenizer.encode(
                stop_char,
                add_special_tokens=False,
            )

            # Only add a separator if it maps to a single
            # tokenizer token. Multi-token separators cannot
            # be caught by this token-level mechanism.
            if len(ids) == 1:

                stop_ids.add(
                    ids[0]
                )

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

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
        print(
            "Newline IDs:",
            sorted(newline_ids)
        )
        print(
            "Stop IDs:",
            sorted(stop_ids)
        )

    # ============================================================
    # AUTOREGRESSIVE DECODING
    # ============================================================

    for step in range(max_new_tokens):

        # --------------------------------------------------------
        # CONDITIONAL FORWARD PASS
        # --------------------------------------------------------

        cond_outputs = model(
            input_ids=cond_ids,
            attention_mask=cond_mask,
        )

        # --------------------------------------------------------
        # UNCONDITIONAL FORWARD PASS
        # --------------------------------------------------------

        uncond_outputs = model(
            input_ids=uncond_ids,
            attention_mask=uncond_mask,
        )

        # --------------------------------------------------------
        # NEXT-TOKEN LOGITS
        # --------------------------------------------------------

        cond_logits = cond_outputs.logits[
            :, -1, :
        ]

        uncond_logits = uncond_outputs.logits[
            :, -1, :
        ]

        # --------------------------------------------------------
        # CLASSIFIER-FREE GUIDANCE
        # --------------------------------------------------------

        guided_logits = classifier_free_guidance(
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            guidance_scale=guidance_scale,
        )

        # --------------------------------------------------------
        # TEMPERATURE
        # --------------------------------------------------------

        guided_logits = (
            guided_logits / temperature
        )

        # ========================================================
        # MINIMUM MOLECULE LENGTH
        # ========================================================
        #
        # Do NOT allow the model to terminate before
        # min_new_tokens have been generated.
        #
        # This masking happens BEFORE softmax.
        # ========================================================

        if step < min_new_tokens:

            if eos_id is not None:

                guided_logits[
                    :, eos_id
                ] = float("-inf")

            for stop_id in stop_ids:

                guided_logits[
                    :, stop_id
                ] = float("-inf")

        # --------------------------------------------------------
        # SOFTMAX
        # --------------------------------------------------------

        probs = F.softmax(
            guided_logits,
            dim=-1,
        )

        # --------------------------------------------------------
        # NaN CHECK
        # --------------------------------------------------------

        if torch.isnan(probs).any():

            raise RuntimeError(
                "NaN detected in sampling probabilities."
            )

        # ========================================================
        # TOP-P / NUCLEUS SAMPLING
        # ========================================================

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

            sorted_mask = (
                cumulative_probs > top_p
            )

            # Keep the first token that crosses top-p.
            sorted_mask[..., 1:] = (
                sorted_mask[..., :-1].clone()
            )

            sorted_mask[..., 0] = False

            sorted_probs = (
                sorted_probs.masked_fill(
                    sorted_mask,
                    0.0,
                )
            )

            # Renormalize.
            sorted_probs = (
                sorted_probs
                /
                sorted_probs.sum(
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

        # ========================================================
        # VERBOSE TOKEN LOG
        # ========================================================

        if verbose:

            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            print(
                f"Step {step:03d} | "
                f"ID={token_id} | "
                f"Token={repr(token_text)}"
            )

        # ========================================================
        # STOP ON EOS
        # ========================================================

        if (
            eos_id is not None
            and token_id == eos_id
        ):

            if verbose:

                print(
                    "\nReached EOS."
                )

            break

        # ========================================================
        # STOP ON NEWLINE / SEPARATOR
        # ========================================================

        if (
            token_id in stop_ids
        ):

            if verbose:

                print(
                    "\nReached stop token. "
                    "Stopping after first molecule."
                )

            break

        # ========================================================
        # APPEND GENERATED TOKEN TO CONDITIONAL BRANCH
        # ========================================================

        cond_ids = torch.cat(
            [
                cond_ids,
                next_token,
            ],
            dim=1,
        )

        # ========================================================
        # APPEND SAME TOKEN TO UNCONDITIONAL BRANCH
        # ========================================================

        uncond_ids = torch.cat(
            [
                uncond_ids,
                next_token,
            ],
            dim=1,
        )

        # ========================================================
        # UPDATE CONDITIONAL ATTENTION MASK
        # ========================================================

        cond_mask = torch.cat(
            [
                cond_mask,
                torch.ones(
                    (
                        cond_mask.size(0),
                        1,
                    ),
                    dtype=cond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        # ========================================================
        # UPDATE UNCONDITIONAL ATTENTION MASK
        # ========================================================

        uncond_mask = torch.cat(
            [
                uncond_mask,
                torch.ones(
                    (
                        uncond_mask.size(0),
                        1,
                    ),
                    dtype=uncond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    # ============================================================
    # EXTRACT ONLY GENERATED TOKENS
    # ============================================================

    generated_ids = (
        cond_ids[0][prompt_len:]
    )

    if verbose:

        print("\nGenerated IDs:")
        print(
            generated_ids.tolist()
        )

    # ============================================================
    # DECODE
    # ============================================================

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if verbose:

        print("\nRaw Decoded Output:")
        print(
            repr(raw_output)
        )

    generated = raw_output

    # ============================================================
    # REMOVE EOS
    # ============================================================

    if tokenizer.eos_token is not None:

        generated = generated.split(
            tokenizer.eos_token
        )[0]

    # ============================================================
    # REMOVE NEWLINE
    # ============================================================

    if stop_newline:

        generated = generated.split(
            "\n"
        )[0]

    # ============================================================
    # DEFENSIVE SEPARATOR CLEANUP
    # ============================================================
    #
    # Token-level stopping should normally catch these.
    # This is a second layer of protection in case a separator
    # is represented by multiple tokenizer tokens or otherwise
    # survives decoding.
    #
    # We keep ONLY the first molecule.
    # ============================================================

    if stop_separators:

        generated = generated.split(
            ";"
        )[0]

        generated = generated.split(
            ","
        )[0]

    # ============================================================
    # FINAL CLEANUP
    # ============================================================

    generated = generated.strip()

    if verbose:

        print("\nFinal Output:")
        print(
            repr(generated)
        )

        print("=" * 80)

    return generated
