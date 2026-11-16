#!/usr/bin/env python3
"""Generate model-written reasoning trajectories and keep only correct answers."""

import argparse
import json
import re
from pathlib import Path


FINAL_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*\(?([A-Z])\)?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_prompt(tokenizer, row):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"({labels[i]}) {option}" for i, option in enumerate(row["options"]))
    content = (
        "Solve the multiple-choice problem. Show your reasoning clearly, then put your "
        "chosen option on the final line exactly as `Final answer: (A)`.\n\n"
        f"Question: {row['question']}\nOptions:\n{options}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer_and_reasoning(text):
    matches = list(FINAL_ANSWER.finditer(text.strip()))
    if not matches:
        return None, ""
    match = matches[-1]
    return match.group(1).upper(), text[:match.start()].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint")
    parser.add_argument("--input", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=1,
                        help="Sample this many trajectories per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-correct-trajectories", type=int,
                        help="Stop after retaining this many correct trajectories")
    args = parser.parse_args()
    if args.num_generations < 1 or args.max_new_tokens < 1:
        parser.error("--num-generations and --max-new-tokens must be positive")
    if args.target_correct_trajectories is not None and args.target_correct_trajectories < 1:
        parser.error("--target-correct-trajectories must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_rows(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Input dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [render_prompt(tokenizer, row) for row in rows]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=False,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    engine = llm.llm_engine
    request_rows = {}
    for index, (row, prompt) in enumerate(zip(rows, prompts)):
        request_id = str(index)
        engine.add_request(request_id, prompt, sampling)
        request_rows[request_id] = row

    from tqdm.auto import tqdm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed = 0
    total = len(rows) * args.num_generations
    progress_total = args.target_correct_trajectories or total
    correct = 0
    with args.output.open("w", encoding="utf-8") as stream, tqdm(
        total=progress_total, desc="Correct trajectories", unit="trajectory"
    ) as progress:
        while engine.has_unfinished_requests():
            for result in engine.step():
                if not result.finished:
                    continue
                row = request_rows[result.request_id]
                for completion in result.outputs:
                    answer, reasoning = extract_answer_and_reasoning(completion.text)
                    if answer is None:
                        continue
                    parsed += 1
                    if answer != row["gold_label"].upper():
                        continue
                    generated = dict(row)
                    generated["rationale"] = reasoning
                    generated["generated_answer"] = answer
                    generated["trajectory_model"] = args.model
                    generated["trajectory_generation"] = completion.text
                    stream.write(json.dumps(generated, ensure_ascii=False) + "\n")
                    stream.flush()
                    correct += 1
                    progress.update(1)
                    if args.target_correct_trajectories and correct >= args.target_correct_trajectories:
                        break
                if args.target_correct_trajectories and correct >= args.target_correct_trajectories:
                    break
            if args.target_correct_trajectories and correct >= args.target_correct_trajectories:
                pending = list(request_rows)
                if pending:
                    engine.abort_request(pending, internal=True)
                break

    print(json.dumps({
        "output": str(args.output), "questions": len(rows), "generations": total,
        "parsed_final_answers": parsed, "correct_trajectories": correct,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
