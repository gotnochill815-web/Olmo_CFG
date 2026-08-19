
import torch
import torch.nn.functional as F

from src.cfg.guidance import classifier_free_guidance


@torch.no_grad()
def cfg_decode_diagnostic(
    model,
    tokenizer,
    conditional_prompt,
    unconditional_prompt,
    guidance_scale=2.0,
    max_new_tokens=128,
    temperature=1.0,
    top_p=0.95,
):

    model.eval()

    device = next(model.parameters()).device

    cond = tokenizer(
        conditional_prompt,
        return_tensors="pt",
        clean_up_tokenization_spaces=False,
    ).to(device)

    uncond = tokenizer(
        unconditional_prompt,
        return_tensors="pt",
        clean_up_tokenization_spaces=False,
    ).to(device)

    cond_ids = cond["input_ids"]
    uncond_ids = uncond["input_ids"]

    cond_mask = cond["attention_mask"]
    uncond_mask = uncond["attention_mask"]

    prompt_len = cond_ids.shape[1]

    eos_id = tokenizer.eos_token_id

    newline_ids = set(
        tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
    )

    diagnostics = []

    for step in range(max_new_tokens):

        cond_logits = model(
            input_ids=cond_ids,
            attention_mask=cond_mask,
        ).logits[:, -1, :]

        uncond_logits = model(
            input_ids=uncond_ids,
            attention_mask=uncond_mask,
        ).logits[:, -1, :]

        guided_logits = classifier_free_guidance(
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            guidance_scale=guidance_scale,
        )

        guided_logits = (
            guided_logits / temperature
        )

        probs = F.softmax(
            guided_logits,
            dim=-1,
        )

        entropy = -(
            probs
            * torch.log(
                probs.clamp_min(1e-12)
            )
        ).sum().item()

        top_prob, top_id = torch.max(
            probs,
            dim=-1,
        )

        eos_prob = (
            probs[0, eos_id].item()
            if eos_id is not None
            else 0.0
        )

        newline_prob = max(
            [
                probs[0, x].item()
                for x in newline_ids
            ],
            default=0.0,
        )

        diagnostics.append({
            "step": step,
            "entropy": entropy,
            "top_token_id": top_id.item(),
            "top_probability": top_prob.item(),
            "eos_probability": eos_prob,
            "newline_probability": newline_prob,
        })

        # Top-p
        if top_p < 1.0:

            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending=True,
                dim=-1,
            )

            cumulative = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            mask = cumulative > top_p

            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_probs = sorted_probs.masked_fill(
                mask,
                0.0,
            )

            sorted_probs = (
                sorted_probs /
                sorted_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )

            sampled = torch.multinomial(
                sorted_probs,
                1,
            )

            next_token = sorted_indices.gather(
                -1,
                sampled,
            )

        else:

            next_token = torch.multinomial(
                probs,
                1,
            )

        token_id = next_token.item()

        if (
            eos_id is not None
            and token_id == eos_id
        ):
            break

        if token_id in newline_ids:
            break

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
                    (1, 1),
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
                    (1, 1),
                    dtype=uncond_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    generated_ids = (
        cond_ids[0][prompt_len:]
    )

    generated = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    generated = generated.split("\n")[0].strip()

    return generated, diagnostics
