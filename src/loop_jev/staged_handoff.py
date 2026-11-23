"""Separate question encoding from option reasoning with a latent-only handoff."""

import torch


def encode_handoff_prompt(tokenizer, question, user_content):
    """Return the ordinary chat prompt and its unpadded option segment start."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    question_text = f"Question: {question}"
    if not user_content.startswith(question_text + "\nOptions:\n"):
        raise ValueError("Expected question and options in the standard prompt layout")
    boundary = text.index(user_content) + len(question_text)
    # Leave any token touching question text on the question side.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (left, right) in enumerate(encoded["offset_mapping"]):
        if boundary <= left < right:
            if text[boundary:left].strip():
                raise ValueError("A token crosses the question and option segments")
            return encoded["input_ids"], i
    raise ValueError("Could not locate option segment")


def handoff_masks(attention_mask, option_start, slot_count, suffix_length):
    """Build SDPA masks, with no path from question to the answer except slots."""
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    first_slot = len(pos) - slot_count - suffix_length
    valid = attention_mask.bool()
    question = (pos < option_start[:, None]) & valid
    options = (pos >= option_start[:, None]) & (pos < first_slot) & valid
    slots = (pos >= first_slot) & (pos < len(pos) - suffix_length)
    answer = pos >= len(pos) - suffix_length
    causal = pos[:, None] >= pos
    write_routes = (
        (question[:, :, None] & question[:, None, :])
        | (options[:, :, None] & options[:, None, :])
        | (slots[:, None] & (question[:, None, :] | slots))
        | (answer[:, None] & (options[:, None, :] | answer))
    )
    read_routes = question[:, :, None] == question[:, None, :]

    def mask(routes):
        # Padding keys stay invisible; padding queries may read only themselves.
        allowed = (routes & causal & valid[:, None, :]) | (~valid[:, :, None] & (pos[:, None] == pos))
        return allowed[:, None]

    return mask(write_routes), mask(read_routes)


def handoff_state(base, written, slot_count, suffix_length, swap=False):
    """Reset everything except the slots; optionally exchange whole slot sets."""
    start, stop = -slot_count - suffix_length, -suffix_length
    slots = written[:, start:stop]
    if swap:
        if slots.shape[0] < 2:
            raise ValueError("Swapping handoff states requires at least two examples per batch")
        slots = slots.roll(1, dims=0)
    return torch.cat((base[:, :start], slots, base[:, stop:]), dim=1)
