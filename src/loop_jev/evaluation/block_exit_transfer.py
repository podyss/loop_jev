"""Inject real teacher updates into a frozen single-pass student on held-out questions."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from loop_jev.evaluation.block_layer_signal import choose_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.training.block_state_cache import cache_metadata
from loop_jev.training.tiny_block_exit_distill import (
    SCORE_DELTA, BlockExitCore, FrozenQwenReadout, batch_inputs, block_exit_basis,
)
from loop_jev.training.tiny_loop_distill import ROUNDS, WIDTH, gather, load_model, teacher_trajectory


STEPS = (1, 5, ROUNDS)


def tokenizer_info(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Expected four single-token labels")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return tokenizer, pad, [ids[0] for ids in labels]


@torch.inference_mode()
def capture_teacher(path, rows, device, layer, batch_size, rank):
    model = load_model(path, device)
    cfg = model.config
    if (cfg.loop_count != ROUNDS or cfg.loop_end_layer != layer or not cfg.latent_slots
            or cfg.loop_schedule != "block" or not cfg.loop_damping):
        raise ValueError("Expected the original damped block teacher")
    enable_configured_looping(model)
    tokenizer, pad, labels = tokenizer_info(path)
    prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in rows]
    weight = model.lm_head.weight[labels]
    batches = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Capture teacher", disable=rank != 0):
        inputs, mask = batch_inputs(prompts, range(start, min(start + batch_size, len(rows))), pad, device)
        updates, scores = teacher_trajectory(model, inputs, mask, weight, projection=None)
        batches.append((updates, scores))
    # The patched forward closes over the model; unpatch before releasing the teacher.
    model.model.forward = model.model._unlooped_forward
    return batches, cfg.hidden_size


def transfer_paths(updates, projection):
    full = updates.cumsum(dim=1)
    projected = (updates @ projection).cumsum(dim=1) @ projection.T
    return full, projected


@torch.inference_mode()
def score_student(path, checkpoint, rows, batches, device, hidden_size, batch_size, rank):
    model = load_model(path, device)
    layer = checkpoint["layer"]
    if (model.config.model_type != "qwen3" or model.config.hidden_size != hidden_size
            or model.config.num_hidden_layers <= layer or getattr(model.config, "loop_count", 1) != 1):
        raise ValueError("Expected the original single-pass student with layers above the injection point")
    tokenizer, pad, labels = tokenizer_info(path)
    readout = FrozenQwenReadout(model, layer, labels)
    prompts = [render_prompt(tokenizer, row) for row in rows]
    projection = block_exit_basis(hidden_size, device)
    core = BlockExitCore(projection).to(device).eval()
    core.load_state_dict(checkpoint["weights"][SCORE_DELTA])
    records = []
    for index, (updates, teacher_scores) in enumerate(tqdm(batches, desc="Inject updates", disable=rank != 0)):
        start = index * batch_size
        inputs, mask = batch_inputs(prompts, range(start, start + len(updates)), pad, device)
        prefix, context = readout.prefix(inputs, mask)
        answer = prefix[:, -1].float()
        full, projected = transfer_paths(updates.to(device), projection)
        learned = core(answer) * checkpoint["delta_scale"]
        columns = {"student_0": readout.scores(prefix, context).tolist()}
        energy = []
        for step in STEPS:
            columns[f"teacher_{step}"] = teacher_scores[:, step - 1].tolist()
            for name, path_updates in (("oracle_full", full), ("oracle_64d", projected), ("learned", learned)):
                state = answer + path_updates[:, step - 1]
                columns[f"{name}_{step}"] = readout.scores(prefix, context, state).tolist()
            energy.append(torch.stack((full[:, step - 1].square().sum(-1),
                                       projected[:, step - 1].square().sum(-1)), dim=-1))
        energies = torch.stack(energy, dim=1).tolist()
        for idx in range(len(updates)):
            row = rows[start + idx]
            records.append(dict(id=row["id"], gold_index=row["gold_index"], energy=energies[idx],
                                scores={name: values[idx] for name, values in columns.items()}))
    return records


def summarize(rows):
    gold = torch.tensor([row["gold_index"] for row in rows])
    scores = {name: torch.tensor([row["scores"][name] for row in rows]) for name in rows[0]["scores"]}
    energy = torch.tensor([row["energy"] for row in rows], dtype=torch.float64).sum(0)
    if not all(values.isfinite().all() for values in (*scores.values(), energy)):
        raise ValueError("Nonfinite diagnostic output")
    correct = {name: values.argmax(-1).eq(gold) for name, values in scores.items()}
    base = correct["student_0"]
    results = {}
    for name, pred in correct.items():
        count = int(pred.sum())
        entry = dict(correct=count, accuracy=count / len(rows))
        if not name.startswith("teacher_"):
            corrected, broke = int((pred & ~base).sum()), int((~pred & base).sum())
            entry.update(corrected=corrected, broke=broke, gain_pp=100 * (corrected - broke) / len(rows))
        results[name] = entry
    return dict(count=len(rows), results=results,
                projection_energy_fraction={str(step): projected / total if total else None
                                            for step, (total, projected) in zip(STEPS, energy.tolist(), strict=True)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "student", "tiny-run", "train", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.examples) < 1:
        parser.error("Expected positive batch size and diagnostic row count")
    teacher = json.loads((args.teacher / "experiment.json").read_text())
    student = json.loads((args.student / "experiment.json").read_text())
    tiny = json.loads((args.tiny_run / "summary.json").read_text())
    checkpoint = torch.load(args.tiny_run / "tiny-loop.pt", map_location="cpu", weights_only=True)
    layer = teacher["loop_end_layer"]
    pool = read_rows(args.train)
    if not pool or any(row.get("split") != "train" or len(row["options"]) != 4 for row in pool):
        parser.error("Expected the original four-option training pool")
    trained = set()
    for experiment in (teacher, student):
        count = experiment["examples"]
        if Path(experiment["train"]).resolve() != args.train.resolve() or not 0 < count <= len(pool):
            parser.error("Training pool must match both frozen models")
        trained.update(row["id"] for row in random.Random(experiment["train_sample_seed"]).sample(pool, count))
    metadata = cache_metadata(train=args.train, model=teacher["model"], teacher=str(args.teacher),
                              layer=layer, seed=teacher["train_sample_seed"], rounds=ROUNDS,
                              damping_step=teacher["loop_damping_step"])
    if (tiny["teacher"] != metadata or Path(checkpoint["student"]).resolve() != args.student.resolve()
            or (checkpoint["layer"], checkpoint["rounds"], checkpoint["width"]) != (layer, ROUNDS, WIDTH)
            or SCORE_DELTA not in checkpoint["weights"] or not 0 < checkpoint["delta_scale"] < float("inf")):
        parser.error("Tiny module does not match the teacher and student for this diagnostic")
    rows = choose_rows(pool, trained, args.examples, args.sample_seed)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if len(rows) < world:
        parser.error("Expected at least one diagnostic example per rank")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    shard = rows[rank::world]
    batches, hidden_size = capture_teacher(args.teacher, shard, device, layer, args.batch_size, rank)
    torch.cuda.empty_cache()
    local = score_student(args.student, checkpoint, shard, batches, device, hidden_size, args.batch_size, rank)
    peak = torch.cuda.max_memory_reserved(device) / 2**30
    if world > 1:
        dist.init_process_group("gloo")
    parts = gather((local, peak), world, rank)
    if world > 1:
        dist.destroy_process_group()
    if rank != 0:
        return
    records = [record for shard_records, _ in parts for record in shard_records]
    by_id = {row["id"]: row for row in records}
    if len(records) != len(rows) or by_id.keys() != {row["id"] for row in rows}:
        raise ValueError("Diagnostic output must cover exactly the selected questions")
    ordered = [by_id[row["id"]] for row in rows]
    report = summarize(ordered)
    report.update(teacher=metadata, student=str(args.student), tiny_run=str(args.tiny_run),
                  sample_seed=args.sample_seed, excluded_training_ids=len(trained),
                  peak_reserved_gib=max(peak for _, peak in parts))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ordered))
    path = args.output / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for name, stats in report["results"].items():
        details = (f"  vs student_0 {stats['gain_pp']:+.2f} pp  +{stats['corrected']}/-{stats['broke']}"
                   if name != "student_0" and "gain_pp" in stats else "")
        print(f"{name:15} {stats['accuracy']:.5f} ({stats['correct']}/{len(rows)}){details}")
    fractions = {step: round(value, 4) if value is not None else None
                 for step, value in report["projection_energy_fraction"].items()}
    print(f"Projection energy fraction: {fractions}")
    print(f"Peak reserved: {report['peak_reserved_gib']:.3f} GiB. Saved: {path}")


if __name__ == "__main__":
    main()
