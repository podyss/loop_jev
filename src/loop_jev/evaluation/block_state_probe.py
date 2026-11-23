"""Patch post-damping states of a frozen block-loop teacher on training rows."""

import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.transition import load_loop_model


def summarize(rows, final):
    names = list(rows[0]["margins"])
    comparisons = {"round5": "round1", final: "round1"}
    for region in ("answer", "workspace"):
        comparisons[f"{region}_late_into_early"] = "round1"
        comparisons[f"{region}_early_into_late"] = final
    result = {}
    for name in names:
        correct = [row["predictions"][name] == row["gold_index"] for row in rows]
        item = {"correct": sum(correct), "accuracy": mean(correct),
                "mean_gold_margin": mean(row["margins"][name] for row in rows)}
        if name in comparisons:
            reference = comparisons[name]
            base = [row["predictions"][reference] == row["gold_index"] for row in rows]
            changes = [row["margins"][name] - row["margins"][reference] for row in rows]
            item.update(reference=reference, delta_pp=100 * (mean(correct) - mean(base)),
                        mean_margin_delta=mean(changes),
                        wrong_to_right=sum(now and not before for now, before in zip(correct, base)),
                        right_to_wrong=sum(before and not now for now, before in zip(correct, base)))
        result[name] = item
    return result


@torch.inference_mode()
def score_batch(model, tokenizer, label_ids, rows):
    cfg = model.config
    device = model.get_input_embeddings().weight.device
    prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in rows]
    suffix = len(tokenizer.encode("Answer: (", add_special_tokens=False))
    regions = {"answer": slice(-1, None), "workspace": slice(-(cfg.latent_slots + suffix), -suffix)}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(rows), max(map(len, prompts))), pad, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompts):
        input_ids[index, -len(ids):] = torch.tensor(ids)
        mask[index, -len(ids):] = 1

    scores = {}
    early = None

    def score(hidden):
        return model.lm_head(hidden[:, -1:, :])[:, 0, label_ids].float()

    def observe(round_number, state, read_suffix):
        nonlocal early
        if round_number == 1:
            early = state.clone()
        if round_number in (1, 5):
            scores[f"round{round_number}"] = score(model.model.norm(read_suffix(state)))
        if round_number == cfg.loop_count:
            for name, region in regions.items():
                for direction, recipient, donor in (
                    ("late_into_early", early, state), ("early_into_late", state, early),
                ):
                    patched = recipient.clone()
                    patched[:, region, :] = donor[:, region, :]
                    scores[f"{name}_{direction}"] = score(model.model.norm(read_suffix(patched)))

    final = model.model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                        use_cache=False, block_state_observer=observe).last_hidden_state
    scores[f"round{cfg.loop_count}"] = score(final)
    scores = {name: value.cpu() for name, value in scores.items()}
    results = []
    for idx, row in enumerate(rows):
        gold = row["gold_index"]
        margins, predictions = {}, {}
        for name, logits in scores.items():
            values = logits[idx, :len(row["options"])]
            margin = values[gold] - torch.cat((values[:gold], values[gold + 1:])).max()
            margins[name] = margin.item()
            predictions[name] = values.argmax().item()
        results.append({"id": row["id"], "gold_index": gold, "margins": margins, "predictions": predictions})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if min(args.max_train_examples, args.batch_size) < 1:
        parser.error("Sample size and batch size must be positive")
    rows = read_rows(args.train)
    if len(rows) > args.max_train_examples:
        indices = random.Random(args.train_sample_seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[index] for index in indices]
    if not rows or any(row.get("split") != "train" or len(row["options"]) not in range(2, 27) for row in rows):
        parser.error("Expected training rows with at least two and at most 26 options")

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={"": device},
                            attn_implementation="sdpa", trust_remote_code=False).eval()
    cfg = model.config
    extras = ("loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
              "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
              "latent_core_width", "option_workspace")
    if (cfg.loop_count < 6 or not cfg.loop_damping or cfg.loop_schedule != "block"
            or not cfg.latent_slots or any(getattr(cfg, name, None) for name in extras)):
        parser.error("Expected a plain damped block-loop teacher with at least six rounds and latent slots")
    enable_configured_looping(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    labels = [tokenizer.encode(chr(ord("A") + i), add_special_tokens=False)
              for i in range(max(len(row["options"]) for row in rows))]
    if any(len(ids) != 1 for ids in labels):
        parser.error("Option labels must be single tokens")
    label_ids = [ids[0] for ids in labels]
    shard = rows[rank::world]
    local = []
    for offset in tqdm(range(0, len(shard), args.batch_size), desc=f"Teacher probe rank {rank}"):
        local.extend(score_batch(model, tokenizer, label_ids, shard[offset:offset + args.batch_size]))
    if world > 1:
        dist.init_process_group("gloo")
        shards = [None] * world if rank == 0 else None
        dist.gather_object(local, shards, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        local = [row for part in shards for row in part]
    by_id = {row["id"]: row for row in local}
    if len(local) != len(rows) or len(by_id) != len(rows):
        raise RuntimeError("Probe output does not cover all sampled IDs exactly once")
    ordered = [by_id[row["id"]] for row in rows]
    results = summarize(ordered, f"round{cfg.loop_count}")
    report = {"model": args.model, "train": str(args.train), "train_sample_seed": args.train_sample_seed,
              "count": len(ordered), "rounds": cfg.loop_count, "damping_step": cfg.loop_damping_step,
              "results": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "probe.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ordered))
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    for name, item in results.items():
        effect = (f"  {item['delta_pp']:+.2f} pp  margin delta {item['mean_margin_delta']:+.3f}"
                  f"  vs {item['reference']}" if "reference" in item else "")
        print(f"{name:27} {item['accuracy']:.5f} ({item['correct']}/{len(ordered)}){effect}")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
