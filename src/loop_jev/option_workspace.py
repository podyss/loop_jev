"""Prompt spans and shared readout for four option-conditioned workspaces."""

import torch


OPTION_COUNT = 4
SLOTS_PER_OPTION = 6
SLOT_COUNT = OPTION_COUNT * SLOTS_PER_OPTION


def encode_option_prompt(tokenizer, row):
    if len(row["options"]) != OPTION_COUNT:
        raise ValueError("Option workspace requires exactly four choices")
    content = f"Question: {row['question']}\nOptions:\n"
    spans = []
    for label, option in zip("ABCD", row["options"]):
        content += f"({label}) "
        start = len(content)
        content += str(option).strip()
        spans.append((start, len(content)))
        content += "\n"
    content = content.removesuffix("\n")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    origin = prompt.index(content)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    token_spans = []
    for start, end in spans:
        tokens = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                  if left < right and start + origin < right and left < end + origin]
        if start == end or not tokens:
            raise ValueError("Could not locate option body in the prompt")
        token_spans.append((tokens[0], tokens[-1] + 1))
    return encoded["input_ids"], token_spans


def pad_option_spans(spans, offsets):
    return torch.tensor(spans, dtype=torch.long) + torch.tensor(offsets, dtype=torch.long)[:, None, None]


def seed_option_slots(hidden, spans, offsets, suffix):
    # Pool option text in FP32; replace only the existing marker slots.
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    mask = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
    weights = mask.float() / mask.sum(dim=-1, keepdim=True).clamp_min(1)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        pooled = (weights @ hidden.float()).to(hidden.dtype)
    slots = (pooled[:, :, None, :] + offsets.to(hidden.dtype)[None, None, :, :]).flatten(1, 2)
    return torch.cat((hidden[:, :-SLOT_COUNT - suffix], slots, hidden[:, -suffix:]), dim=1)
