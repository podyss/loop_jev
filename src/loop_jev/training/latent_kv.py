"""Offline CoT KV anchors for a fixed-length latent answer workspace.

Anchors use normalized keys before RoPE and values before attention, at one
decoder layer. They are averaged into fixed-width bins over rationale tokens.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def marker_token_id(tokenizer):
    ids = tokenizer.encode("?", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("Latent workspace marker must be a single token")
    return ids[0]


def append_workspace(prompt_ids, tokenizer, slots):
    return (prompt_ids + [marker_token_id(tokenizer)] * slots
            + tokenizer.encode("Answer: (", add_special_tokens=False))


def compressed_kv(key, value, slots):
    """Pool contiguous rationale positions, retaining the per-head geometry."""
    if key.ndim != 3 or value.shape != key.shape or key.shape[0] < 1:
        raise ValueError("Expected nonempty, matching [tokens, kv_heads, head_dim] tensors")
    if slots < 1:
        raise ValueError("slots must be positive")
    def pool(tensor):
        tokens, heads, width = tensor.shape
        packed = tensor.permute(1, 2, 0).reshape(1, heads * width, tokens).float()
        return F.adaptive_avg_pool1d(packed, slots).reshape(heads, width, slots).permute(2, 0, 1)
    return pool(key), pool(value)


def kv_alignment_loss(student_key, student_value, target_key, target_value):
    """Mean cosine loss across positions/heads and K/V, computed in FP32."""
    if student_key.shape != target_key.shape or student_value.shape != target_value.shape:
        raise ValueError("Student and teacher KV shapes must match")
    return ((1 - F.cosine_similarity(student_key.float(), target_key.float(), dim=-1)).mean()
            + (1 - F.cosine_similarity(student_value.float(), target_value.float(), dim=-1)).mean()) / 2


def capture_workspace_kv(model, layer_index, slots, answer_length):
    """Capture only latent positions from the last visit to a recurrent layer."""
    attention = model.model.layers[layer_index].self_attn
    captured = {}

    def capture_key(_module, _inputs, output):
        captured["key"] = output[:, -slots - answer_length:-answer_length, :, :]

    def capture_value(_module, _inputs, output):
        values = output[:, -slots - answer_length:-answer_length, :]
        captured["value"] = values.reshape(values.shape[0], slots, -1, attention.head_dim)

    hooks = [attention.k_norm.register_forward_hook(capture_key),
             attention.v_proj.register_forward_hook(capture_value)]
    return captured, hooks


def load_cache(path, *, train, model, slots, layer, max_length, seed, selected_ids):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "train_sha256": hashlib.sha256(Path(train).read_bytes()).hexdigest(),
        "model": model,
        "slots": slots,
        "layer": layer,
        "max_length": max_length,
        "seed": seed,
    }
    for name, value in expected.items():
        if cache.get(name) != value:
            raise ValueError(f"KV cache {name} differs: expected {value!r}, got {cache.get(name)!r}")
    if set(cache["targets"]) != set(selected_ids):
        raise ValueError("KV cache must contain exactly the training examples")
    return cache["targets"]


def main():
    parser = argparse.ArgumentParser(description="Precompute compressed rationale KV targets")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--latent-slots", type=int, default=24)
    parser.add_argument("--kv-layer", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.latent_slots < 1 or args.max_length < 1:
        parser.error("Slots and max length must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing existing cache: {args.output}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm
    from loop_jev.evaluation.zero_shot import render_prompt
    from loop_jev.training.answer_slot_distill import read_rows, remove_answer_disclosure

    rows = read_rows(args.train)
    if args.max_train_examples and len(rows) > args.max_train_examples:
        selected = random.Random(args.seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[index] for index in selected]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
        device_map={"": 0},
    ).eval()
    if not 0 <= args.kv_layer < model.config.num_hidden_layers:
        parser.error("kv-layer is outside the decoder")
    attention = model.model.layers[args.kv_layer].self_attn
    captured = {}
    hooks = [
        attention.k_norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("key", out)),
        attention.v_proj.register_forward_hook(lambda _m, _i, out: captured.__setitem__("value", out)),
    ]
    targets = {}
    try:
        for row in tqdm(rows, desc="CoT KV targets", unit="example"):
            prompt = render_prompt(tokenizer, row)
            answer_length = len(tokenizer.encode("Answer: (", add_special_tokens=False))
            prompt = prompt[:-answer_length]
            rationale = remove_answer_disclosure(row.get("rationale", ""))
            prefix = tokenizer.encode("Reasoning: ", add_special_tokens=False)
            rationale_ids = tokenizer.encode(rationale, add_special_tokens=False)
            room = args.max_length - len(prompt) - len(prefix)
            if room < 1:
                raise ValueError(f"No room for a rationale for {row['id']}")
            rationale_ids = rationale_ids[:room]
            if not rationale_ids:
                raise ValueError(f"Empty rationale for {row['id']}")
            ids = torch.tensor([prompt + prefix + rationale_ids], device="cuda:0")
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
            key = captured.pop("key")[0, -len(rationale_ids):].float()
            value = captured.pop("value")[0, -len(rationale_ids):].reshape(
                len(rationale_ids), -1, attention.head_dim,
            ).float()
            pooled_key, pooled_value = compressed_kv(key, value, args.latent_slots)
            targets[row["id"]] = (pooled_key.to("cpu", dtype=torch.bfloat16),
                                   pooled_value.to("cpu", dtype=torch.bfloat16))
    finally:
        for hook in hooks:
            hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "model": args.model, "slots": args.latent_slots, "layer": args.kv_layer,
        "max_length": args.max_length, "seed": args.seed, "targets": targets,
    }, args.output)
    print(json.dumps({"cache": str(args.output), "examples": len(targets),
                      "target_shape": list(next(iter(targets.values()))[0].shape)}))


if __name__ == "__main__":
    main()
