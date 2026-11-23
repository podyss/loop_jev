"""Build checked question interventions and train on paired answer margins."""

import argparse
import json
import random
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import torch
from torch.nn import functional as F

from loop_jev.training.program_composition import CALCULATE, NUMBER, _verified_trace


SMALL_NUMBERS = ("zero", "one", "two|half|twice|double|pair", "three|thrice|triple|third",
                 "four|quarter|fourth", "five|fifth", "six|sixth", "seven", "eight", "nine", "ten")


def _fraction_text(value):
    return str(value) if value.denominator == 1 else f"({value})"


def _recompute(steps, original, replacement):
    values, expressions = [], []

    def evaluate(node):
        op, value, *tail = node
        if op == "ref":
            return values[value], _fraction_text(values[value])
        if op == "literal":
            value = replacement if value == original else value
            return value, _fraction_text(value)
        if op == "ambiguous":
            raise ValueError("Ambiguous calculator reference")
        left, left_text = evaluate(value)
        right, right_text = evaluate(tail[0])
        return CALCULATE[op](left, right), f"({left_text}{op}{right_text})"

    for step in steps:
        value, expression = evaluate(step)
        values.append(value)
        expressions.append(expression)
    return values, expressions


def make_pair(row, rng):
    trace = _verified_trace(row) if len(row["options"]) == 4 else None
    if trace is None:
        return None
    steps, old_values, inputs, literals = trace
    question = row["question"]
    if any("=" in line and "<<" not in line and re.search(r"\d", line)
           for line in row["rationale"].splitlines()):
        return None
    # A precomputed number missing from the annotations can depend on the edit.
    named = {i for i, words in enumerate(SMALL_NUMBERS) if re.search(rf"\b(?:{words})\b", question, re.I)}
    if any(value not in inputs and value not in named and value != 1 for value in literals):
        return None
    matches = list(NUMBER.finditer(question))
    old_answer = old_values[-1]
    try:
        options = [Fraction(str(option).replace(",", "")) for option in row["options"]]
    except (ValueError, ZeroDivisionError):
        return None
    if old_answer.denominator != 1 or old_answer <= 0 or len(set(options)) != 4:
        return None
    rng.shuffle(matches)
    for match in matches:
        text = match[0]
        before, after = question[:match.start()], question[match.end():]
        if (not text.isascii() or not text.isdigit() or re.search(r"[\w.,/:+\-]$", before)
                or re.match(r"[\w.,:/%+\-]", after)):
            continue
        # Avoid percentages, clock times and ages; prices and counts stay in their original units.
        if (not before.endswith("$") and not re.match(r" (?:[a-z]+s\b|people\b|children\b)", after, re.I)):
            continue
        if (re.match(r" (?:a\.m\.|p\.m\.|years? old|(?:days?|years?) ago|(?:minutes?|hours?) later)", after, re.I)
                or re.search(r"\b(?:aged|turned|age of|at age)\s+$", before, re.I)):
            continue
        original = Fraction(text)
        if not 3 <= original <= 500 or inputs[original] != 1 or literals[original] != 1:
            continue
        deltas = [-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]
        rng.shuffle(deltas)
        for delta in deltas:
            replacement = original + delta
            if not 2 <= replacement <= 500 or replacement in inputs:
                continue
            try:
                values, expressions = _recompute(steps, original, replacement)
            except (ValueError, ZeroDivisionError):
                continue
            new_answer = values[-1]
            if (new_answer == old_answer or new_answer.denominator != 1 or new_answer < 1
                    or any(new < 0 or abs(new) > 1_000_000 or (old.denominator == 1 and new.denominator != 1)
                           for old, new in zip(old_values, values, strict=True))):
                continue
            shared = list(row["options"])
            if new_answer not in options:
                distractor = rng.choice([i for i, option in enumerate(options) if option != old_answer])
                shared[distractor] = str(new_answer)
            rng.shuffle(shared)
            label = {Fraction(str(option).replace(",", "")): i for i, option in enumerate(shared)}
            paired = []
            for variant, prompt, answer, rationale in (
                ("original", question, old_answer, row["rationale"]),
                ("changed", before + str(replacement) + after, new_answer,
                 "\n".join(f"{expr} = {_fraction_text(value)}" for expr, value in zip(expressions, values))),
            ):
                index = label[answer]
                paired.append(dict(row, id=f"{row['id']}:{variant}", pair_id=row["id"], variant=variant,
                                   question=prompt, options=shared, gold_index=index,
                                   gold_label=chr(ord("A") + index), rationale=rationale))
            return paired
    return None


def pair_groups(rows):
    pairs, single = defaultdict(list), []
    for index, row in enumerate(rows):
        if row.get("pair_id") is None:
            single.append(index)
        else:
            pairs[row["pair_id"]].append(index)
    for indices in pairs.values():
        members = [rows[i] for i in indices]
        if (len(members) != 2 or {row["variant"] for row in members} != {"original", "changed"}
                or members[0]["options"] != members[1]["options"]
                or members[0]["gold_index"] == members[1]["gold_index"]):
            raise ValueError(f"Invalid pair: {members[0]['pair_id']}")
    if not pairs or len(single) % 2:
        raise ValueError("Pair training needs at least one pair and an even number of single examples")
    return list(pairs.values()) + [single[i:i + 2] for i in range(0, len(single), 2)]


def paired_margin_loss(logits, targets, valid):
    size = logits.shape[0]
    if size % 2 or valid.shape != (size // 2,):
        raise ValueError("Expected adjacent pairs and one validity flag per pair")
    # The same contrast must favor opposite options on the two versions.
    other = targets.reshape(-1, 2).flip(1).reshape(-1)
    margin = logits.gather(1, targets[:, None]) - logits.gather(1, other[:, None])
    both = margin.reshape(-1, 2)
    penalty = F.softplus(1.0 - both.min(dim=1).values)
    loss = (penalty * valid).sum() / valid.numel()
    stats = torch.stack(((penalty.detach() * valid).sum(), ((both.min(dim=1).values > 0) & valid).sum(), valid.sum()))
    return loss, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation-input", "--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--pairs", type=int, default=128)
    args = parser.parse_args()
    from loop_jev.training.answer_slot_distill import read_rows

    if not 0 < 2 * args.pairs <= args.examples or args.examples % 2:
        parser.error("Require an even total and space for every pair")
    rows = random.Random(42).sample(read_rows(args.train), args.examples)
    evaluation = read_rows(args.evaluation_input)
    if {row["id"] for row in rows} & {row["id"] for row in evaluation}:
        raise ValueError("Training and evaluation have shared source IDs")
    rng = random.Random(854)
    candidates = [pair for row in rows if (pair := make_pair(row, rng))]
    if len(candidates) < args.pairs:
        raise ValueError(f"Only {len(candidates)} eligible training interventions")
    paired = candidates[:args.pairs]
    used = {pair[0]["pair_id"] for pair in paired}
    single = [row for row in rows if row["id"] not in used][:args.examples - 2 * args.pairs]
    training = single + [row for pair in paired for row in pair]
    diagnostics = [row for original in evaluation if (pair := make_pair(original, rng)) for row in pair]
    if not diagnostics:
        raise ValueError("No eligible diagnostic pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", training), ("pairs", diagnostics)):
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Training: {len(training)} rows ({len(paired)}/{len(candidates)} eligible pairs used); "
          f"evaluation: {len(diagnostics) // 2} held-out pairs")


if __name__ == "__main__":
    main()
