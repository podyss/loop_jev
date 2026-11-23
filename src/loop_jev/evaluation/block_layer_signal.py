"""Diagnose frozen teacher layer interventions on held-out training-pool questions."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping, without_looping
from loop_jev.training.block_layer_effect import extract_scores
from loop_jev.training.block_state_cache import cache_metadata
from loop_jev.transition import load_loop_model


def choose_rows(rows, trained_ids, count, seed):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in the diagnostic source")
    seen_questions = {row["question"].strip() for row in rows if row["id"] in trained_ids}
    pool = [row for row in rows if row["id"] not in trained_ids and row["question"].strip() not in seen_questions]
    if not 0 < count <= len(pool):
        raise ValueError(f"Need {count} diagnostic rows, found {len(pool)} after exclusion")
    return random.Random(seed).sample(pool, count)


def summarize(rows, layers):
    gold = torch.tensor([row["gold_index"] for row in rows])
    scores = {key: torch.tensor([row["scores"][key] for row in rows]) for key in rows[0]["scores"]}
    if not all(score.isfinite().all() for score in scores.values()):
        raise ValueError("Nonfinite option scores in the diagnostic")
    correct = {key: score.argmax(1).eq(gold) for key, score in scores.items()}
    gained = ~correct["bypass"] & correct["full"]
    harmed = correct["bypass"] & ~correct["full"]
    gold_mask = torch.nn.functional.one_hot(gold, 4).bool()

    def margin(score):
        return score[gold_mask] - score.masked_fill(gold_mask, -torch.inf).amax(1)

    def average(values):
        return values.mean().item() if values.numel() else None

    report = {"count": len(rows), "bypass_correct": int(correct["bypass"].sum()),
              "full_correct": int(correct["full"].sum()), "loop_corrected": int(gained.sum()),
              "loop_broke": int(harmed.sum()), "layers": {}}
    for layer in layers:
        name = f"patch_{layer}"
        change = margin(scores["full"]) - margin(scores[name])
        effect = scores["full"] - scores[name]
        effect -= effect.mean(1, keepdim=True)
        report["layers"][str(layer)] = {
            "correct": int(correct[name].sum()),
            "corrected_questions_lost": int((gained & ~correct[name]).sum()),
            "other_full_correct_lost": int((correct["full"] & ~gained & ~correct[name]).sum()),
            "full_wrong_recovered": int((~correct["full"] & correct[name]).sum()),
            "mean_margin_drop_on_corrected": average(change[gained]),
            "corrected_with_margin_drop": int((gained & (change > 0)).sum()),
            "centered_score_effect_rms": effect.square().mean().sqrt().item(),
        }
    return report


@torch.inference_mode()
def score_batch(model, tokenizer, rows, layers, label_ids):
    cfg = model.config
    device = model.get_input_embeddings().weight.device
    prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in rows]
    width = max(map(len, prompts))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = torch.tensor([[pad] * (width - len(prompt)) + prompt for prompt in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts], device=device)
    full, patched = extract_scores(model, inputs, mask, layers, label_ids)
    with without_looping(model):
        hidden = model.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
        bypass = model.lm_head(hidden[:, -1:, :])[:, 0, label_ids].float()
    columns = {"bypass": bypass.tolist(), "full": full.tolist()}
    columns.update({f"patch_{layer}": patched[:, idx].tolist() for idx, layer in enumerate(layers)})
    return [{"id": row["id"], "gold_index": row["gold_index"],
             "scores": {name: values[idx] for name, values in columns.items()}}
            for idx, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "train", "train-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    experiment = json.loads((args.teacher / "experiment.json").read_text())
    cache = torch.load(args.train_cache, map_location="cpu", weights_only=True)
    metadata = cache_metadata(train=args.train, model=experiment["model"], teacher=str(args.teacher),
                              layer=experiment["loop_end_layer"], seed=experiment["train_sample_seed"],
                              rounds=experiment["loop_count"], damping_step=experiment["loop_damping_step"])
    rows = read_rows(args.train)
    if not rows or any(row.get("split") != "train" or len(row["options"]) != 4 for row in rows):
        parser.error("Expected four-option training-pool rows")
    count = experiment["examples"]
    indices = (random.Random(metadata["seed"]).sample(range(len(rows)), count)
               if count < len(rows) else range(len(rows)))
    trained_ids = {rows[index]["id"] for index in indices}
    if (cache["metadata"] != metadata or len(trained_ids) != count or trained_ids != set(cache["ids"])
            or Path(experiment["train"]).resolve() != args.train.resolve()):
        parser.error("Training pool or exclusion cache does not match the frozen teacher")
    rows = choose_rows(rows, trained_ids, args.examples, args.sample_seed)
    layers = cache["layers"]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=False)
    model = load_loop_model(str(args.teacher), dtype=torch.bfloat16, device_map={"": device},
                            attn_implementation="sdpa", trust_remote_code=False).eval()
    enable_configured_looping(model)
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(tokens) != 1 for tokens in label_tokens):
        parser.error("Expected single-token options")
    label_ids = [tokens[0] for tokens in label_tokens]
    local, shard = [], rows[rank::world]
    for offset in tqdm(range(0, len(shard), args.batch_size), desc=f"Layer probe rank {rank}"):
        local.extend(score_batch(model, tokenizer, shard[offset:offset + args.batch_size], layers, label_ids))
    if world > 1:
        dist.init_process_group("gloo")
        parts = [None] * world if rank == 0 else None
        dist.gather_object(local, parts, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        local = [item for part in parts for item in part]
    results = {row["id"]: row for row in local}
    if len(local) != len(rows) or results.keys() != {row["id"] for row in rows}:
        raise ValueError("Diagnostic output does not match the selected questions")
    ordered = [results[row["id"]] for row in rows]
    report = summarize(ordered, layers)
    report.update(teacher=str(args.teacher), sample_seed=args.sample_seed, excluded_training_ids=count,
                  data=str(args.train), loop_count=metadata["rounds"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ordered))
    path = args.output / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    corrected = report["loop_corrected"]
    print(f"Bypass: {report['bypass_correct']}/{len(rows)}; full: {report['full_correct']}/{len(rows)}; "
          f"loop corrected: {corrected}; loop broke: {report['loop_broke']}")
    for layer, stats in report["layers"].items():
        drop = stats["mean_margin_drop_on_corrected"]
        delta = "n/a" if drop is None else f"{drop:+.4f}"
        print(f"Layer {layer}: {stats['correct']}/{len(rows)}; lost {stats['corrected_questions_lost']}/{corrected} "
              f"loop corrections; corrected-set mean margin drop {delta}; "
              f"RMS {stats['centered_score_effect_rms']:.4f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
