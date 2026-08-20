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


def block_repeated_ngrams(logits, generated_ids, n=4):
    """
    Prevent the decoder from re-entering a loop by banning
    a token that would recreate an n-gram already seen.
    """
    if len(generated_ids) < n - 1:
        return logits

    current_ngram = tuple(generated_ids[-(n - 1):])

    for i in range(len(generated_ids) - (n - 1)):
        if tuple(generated_ids[i:i + n - 1]) == current_ngram:
            banned_token = generated_ids[i + n - 1]
            logits[:, banned_token] = float("-inf")

    return logits


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
    block_repeats: bool = True,      # NEW
    repeat_ngram_size: int = 4,      # NEW
    balance_parens: bool = True,     # NEW
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
        print(f"Block repeats  : {block_repeats}")
        print(f"Balance parens : {balance_parens}")
        print("=" * 80)

    cond_inputs = tokenizer(conditional_prompt, return_tensors="pt").to(device)
    uncond_inputs = tokenizer(unconditional_prompt, return_tensors="pt").to(device)

    cond_ids = cond_inputs["input_ids"]
    cond_mask = cond_inputs["attention_mask"]
    uncond_ids = uncond_inputs["input_ids"]
    uncond_mask = uncond_inputs["attention_mask"]

    prompt_len = cond_ids.shape[1]
    eos_id = tokenizer.eos_token_id

    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))

    stop_ids = set(newline_ids)
    if stop_separators:
        for stop_char in [";", ","]:
            ids = tokenizer.encode(stop_char, add_special_tokens=False)
            if len(ids) == 1:
                stop_ids.add(ids[0])

    # ------------------------------------------------------------
    # PARENTHESIS TOKEN IDS
    # ------------------------------------------------------------
    open_paren_ids = set()
    close_paren_ids = set()
    if balance_parens:
        for char in ["(", ")"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            if len(ids) == 1:
                (open_paren_ids if char == "(" else close_paren_ids).add(ids[0])
        # NOTE: your tokenizer merges multi-char tokens like '(' combined with
        # other atoms (e.g. earlier trace showed 'OC1C(' as a single unit is NOT
        # the case here, but some tokens like 'C(' or ')C' may exist as one
        # piece — if so, also scan the vocab for tokens CONTAINING '(' or ')'
        # and count their net paren contribution per token, not just exact
        # single-char matches. Simple version below only handles exact match.

    if verbose:
        print("\nConditional IDs:")
        print(cond_ids)
        print("\nConditional Tokens:")
        print(tokenizer.convert_ids_to_tokens(cond_ids[0]))
        print("\nUnconditional IDs:")
        print(uncond_ids)
        print("\nUnconditional Tokens:")
        print(tokenizer.convert_ids_to_tokens(uncond_ids[0]))
        print("\nEOS ID:", eos_id)
        print("Newline IDs:", sorted(newline_ids))

    generated_ids_list = []  # track plain python ints for ngram/paren logic
    open_paren_count = 0

    for step in range(max_new_tokens):

        cond_outputs = model(input_ids=cond_ids, attention_mask=cond_mask)
        uncond_outputs = model(input_ids=uncond_ids, attention_mask=uncond_mask)

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

        if step < min_new_tokens:
            if eos_id is not None:
                guided_logits[:, eos_id] = float("-inf")
            for stop_id in stop_ids:
                guided_logits[:, stop_id] = float("-inf")

        # ---------------------------------------------------------
        # BLOCK REPEATED N-GRAMS  (NEW)
        # ---------------------------------------------------------
        if block_repeats:
            guided_logits = block_repeated_ngrams(
                guided_logits, generated_ids_list, n=repeat_ngram_size
            )

        # ---------------------------------------------------------
        # BLOCK ')' IF NO OPEN '(' AVAILABLE  (NEW)
        # ---------------------------------------------------------
        if balance_parens and open_paren_count <= 0:
            for cid in close_paren_ids:
                guided_logits[:, cid] = float("-inf")

        probs = F.softmax(guided_logits, dim=-1)

        if torch.isnan(probs).any():
            raise RuntimeError("NaN detected in sampling probabilities.")

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()

        if verbose:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"Step {step:03d} | ID={token_id} | Token={repr(token_text)}")

        if eos_id is not None and token_id == eos_id:
            if verbose:
                print("\nReached EOS.")
            break

        if stop_newline and token_id in stop_ids:
            if verbose:
                print("\nReached stop token. Stopping after first molecule.")
            break

        # update tracking state
        generated_ids_list.append(token_id)
        if balance_parens:
            if token_id in open_paren_ids:
                open_paren_count += 1
            elif token_id in close_paren_ids:
                open_paren_count -= 1

        cond_ids = torch.cat([cond_ids, next_token], dim=1)
        uncond_ids = torch.cat([uncond_ids, next_token], dim=1)

        cond_mask = torch.cat([cond_mask, torch.ones((cond_mask.size(0), 1), dtype=cond_mask.dtype, device=device)], dim=1)
        uncond_mask = torch.cat([uncond_mask, torch.ones((uncond_mask.size(0), 1), dtype=uncond_mask.dtype, device=device)], dim=1)

    generated_ids = cond_ids[0][prompt_len:]

    if verbose:
        print("\nGenerated IDs:")
        print(generated_ids.tolist())

    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)

    if verbose:
        print("\nRaw Decoded Output:")
        print(repr(raw_output))

    generated = raw_output
    if tokenizer.eos_token is not None:
        generated = generated.split(tokenizer.eos_token)[0]
    if stop_newline:
        generated = generated.split("\n")[0]
    if stop_separators:
        generated = generated.split(";")[0]
        generated = generated.split(",")[0]

    generated = generated.strip()

    if verbose:
        print("\nFinal Output:")
        print(repr(generated))
        print("=" * 80)

    return generated
