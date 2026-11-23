"""Offline-only: test how much of a later update lies in the early update span."""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


def project(path, target):
    base = path[-1].float()
    basis = torch.stack([b.float() - a.float() for a, b in zip(path[:-1], path[1:], strict=True)], dim=1)
    basis = basis.flatten(2)
    desired = (target.float() - base).flatten(1).unsqueeze(-1)
    scale = basis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    basis = basis / scale
    gram = basis @ basis.transpose(1, 2)
    weights = torch.linalg.solve(gram + 1e-4 * torch.eye(basis.shape[1], device=basis.device), basis @ desired)
    update = (basis.transpose(1, 2) @ weights).reshape_as(base)
    return update


@torch.inference_mode()
def run(model_path, input_path, limit):
    rows = random.Random(42).sample(read_rows(input_path), limit)
    model = load_loop_model(model_path, dtype=torch.bfloat16, device_map={"": 0},
                            attn_implementation="sdpa", trust_remote_code=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    cfg = model.config
    marker = marker_token_id(tokenizer)
    validate_checkpoint(cfg, marker)
    enable_configured_looping(model)
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Expected one token per answer label")
    label_ids = torch.tensor([ids[0] for ids in labels], device=model.device)
    head = model.lm_head.weight[label_ids]
    width = cfg.latent_slots + len(tokenizer.encode("Answer: (", add_special_tokens=False))
    names = ("round3", "round10", "oracle_span", "oracle_span_split", "orthogonal_only")
    records = []
    for row in tqdm(rows, desc="Early trajectory span"):
        prompt = render_prompt(tokenizer, row, cfg.latent_slots, marker)
        ids = torch.tensor([prompt], device=model.device)
        path, prediction = [], []
        energies = None

        def observe(round_no, before, state, suffix, _replay):
            nonlocal energies
            if round_no <= 3:
                if round_no == 1:
                    path.append(before)
                path.append(state)
            elif round_no == 10:
                base = path[-1].float()
                update = project(path, state)
                split_update = torch.cat(tuple(project([x[:, region] for x in path], state[:, region])
                                               for region in (slice(None, -width), slice(-width, None))), dim=1)
                residual = state.float() - base - update
                candidate = torch.cat((path[-1], (base + update).to(state.dtype),
                                       (base + split_update).to(state.dtype),
                                       (base + residual).to(state.dtype)), dim=0)
                # The suffix closure shares the one-prompt causal mask and RoPE across candidates.
                scores = (model.model.norm(suffix(candidate)[:, -1]) @ head.T).argmax(dim=-1)
                prediction.extend(scores.tolist())
                desired = state.float() - base
                energies = (1 - torch.stack((residual.square().sum(), (desired - split_update).square().sum()))
                            / desired.square().sum().clamp_min(1e-9)).tolist()

        output = model(ids, use_cache=False, logits_to_keep=1, block_round_observer=observe)
        scores = prediction[:1] + [output.logits[0, -1, label_ids].argmax().item()] + prediction[1:]
        records.append(dict(id=row["id"], gold=row["gold_index"], pred=dict(zip(names, scores, strict=True)),
                            energy=dict(zip(names[2:4], energies, strict=True))))
    return dict(model=model_path, input=str(input_path), total=len(rows),
                correct={key: sum(row["pred"][key] == row["gold"] for row in records) for key in names},
                mean_explained_energy={key: sum(row["energy"][key] for row in records) / len(rows)
                                       for key in names[2:4]}, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("Limit must be positive")
    report = run(args.model, args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
