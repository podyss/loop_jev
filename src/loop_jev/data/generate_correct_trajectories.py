#!/usr/bin/env python3
"""Generate beam-search reasoning trajectories and keep only correct answers."""

import argparse
import json
import multiprocessing as mp
import os
import re
from pathlib import Path


FINAL_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*\(?([A-Z])\)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
VLLM_TASK_WINDOW_SIZE = 1
TRANSFORMERS_BATCH_SIZE = 4


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
        enable_thinking=True,
    )


def extract_answer_and_reasoning(text):
    matches = list(FINAL_ANSWER.finditer(text.strip()))
    if not matches:
        return None, ""
    match = matches[-1]
    return match.group(1).upper(), text[:match.start()].strip()


def _vllm_dp_worker(rank, gpu_id, task_queue, result_queue, model_name, max_new_tokens,
                    max_model_len, gpu_memory_utilization):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    try:
        from vllm import LLM
        from vllm.sampling_params import BeamSearchParams

        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=42,
        )
        params = BeamSearchParams(beam_width=4, max_tokens=max_new_tokens, temperature=0.0)
        tokenizer = llm.get_tokenizer()
        result_queue.put(("ready", rank, None, None))
    except BaseException as exc:
        result_queue.put(("error", rank, None, repr(exc)))
        return

    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(("stopped", rank, None, None))
            return
        batch_id, prompts = task
        try:
            outputs = llm.beam_search(prompts, params, use_tqdm=False)
            candidates = []
            for output in outputs:
                prompt_length = len(output.sequences[0].orig_prompt["prompt_token_ids"])
                candidates.append([
                    tokenizer.decode(sequence.tokens[prompt_length:], skip_special_tokens=True)
                    for sequence in output.sequences
                ])
            result_queue.put(("done", rank, batch_id, candidates))
        except BaseException as exc:
            result_queue.put(("error", rank, batch_id, repr(exc)))
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint")
    parser.add_argument("--input", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=1,
                        help="Return this many beam-search candidates per question")
    parser.add_argument("--num-beams", type=int,
                        help="Beam width (defaults to --num-generations)")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--data-parallel-size", type=int, default=2,
                        help="Independent one-GPU vLLM workers")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-correct-trajectories", type=int,
                        help="Stop after retaining this many correct trajectories")
    args = parser.parse_args()
    if args.num_generations < 1 or args.max_new_tokens < 1 or args.max_model_len < 1:
        parser.error("--num-generations, --max-new-tokens, and --max-model-len must be positive")
    num_beams = args.num_beams or args.num_generations
    if num_beams < args.num_generations:
        parser.error("--num-beams must be greater than or equal to --num-generations")
    if num_beams < 2:
        parser.error("beam search requires at least two beams")
    if args.data_parallel_size < 1:
        parser.error("--data-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.backend == "vllm" and num_beams != 4:
        parser.error("The vLLM beam-search backend currently requires --num-beams 4")
    if args.target_correct_trajectories is not None and args.target_correct_trajectories < 1:
        parser.error("--target-correct-trajectories must be positive")
    from transformers import AutoTokenizer

    rows = read_rows(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Input dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if args.backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        model.eval()
        input_device = model.get_input_embeddings().weight.device

    from tqdm.auto import tqdm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed = 0
    generated_count = 0
    saved_count = 0
    total = len(rows) * args.num_generations
    progress_total = args.target_correct_trajectories or total
    correct = 0
    completed_windows = 0
    window_total = 0
    with args.output.open("w", encoding="utf-8") as stream, tqdm(
        total=progress_total, desc="Successful trajectories", unit="success"
    ) as progress:
        def process_candidates(row, candidate_texts):
            nonlocal parsed, generated_count, saved_count, correct
            for text in candidate_texts[:args.num_generations]:
                generated_count += 1
                answer, reasoning = extract_answer_and_reasoning(text)
                if answer is None:
                    is_correct = False
                else:
                    parsed += 1
                    is_correct = answer == row["gold_label"].upper()
                if is_correct:
                    correct += 1
                    if not args.target_correct_trajectories or saved_count < args.target_correct_trajectories:
                        generated = dict(row)
                        generated["rationale"] = reasoning
                        generated["generated_answer"] = answer
                        generated["trajectory_model"] = args.model
                        generated["trajectory_generation"] = text
                        generated["trajectory_decoding"] = f"{args.backend}_beam_search"
                        generated["trajectory_num_beams"] = num_beams
                        if args.backend == "vllm":
                            generated["trajectory_data_parallel_size"] = args.data_parallel_size
                        stream.write(json.dumps(generated, ensure_ascii=False) + "\n")
                        stream.flush()
                        saved_count += 1
                        progress.update(1)
                rate = correct / generated_count if generated_count else 0.0
                progress.set_postfix(
                    generated=generated_count,
                    correct=saved_count,
                    success_rate=f"{rate:.1%}",
                    windows=f"{completed_windows}/{window_total}",
                )
        prompts = [render_prompt(tokenizer, row) for row in rows]
        prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
        for row, prompt_length in zip(rows, prompt_lengths):
            if prompt_length + args.max_new_tokens > args.max_model_len:
                raise ValueError(
                    f"Prompt plus max-new-tokens ({prompt_length} + {args.max_new_tokens}) "
                    f"exceeds --max-model-len {args.max_model_len} for {row.get('id', '<unknown>')}"
                )

        if args.backend == "vllm":
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpu_ids = [device.strip() for device in visible_devices.split(",") if device.strip()]
            if not gpu_ids:
                raise ValueError("Set CUDA_VISIBLE_DEVICES to the GPUs for vLLM data parallelism")
            if len(gpu_ids) < args.data_parallel_size:
                raise ValueError(
                    f"Requested data-parallel-size {args.data_parallel_size}, "
                    f"but only {len(gpu_ids)} GPUs are visible"
                )

            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            task_queues = [ctx.Queue() for _ in range(args.data_parallel_size)]
            workers = [
                ctx.Process(
                    target=_vllm_dp_worker,
                    args=(rank, gpu_ids[rank], task_queues[rank], result_queue, args.model,
                          args.max_new_tokens, args.max_model_len, args.gpu_memory_utilization),
                )
                for rank in range(args.data_parallel_size)
            ]
            for worker in workers:
                worker.start()

            batches = [
                (start, prompts[start:start + VLLM_TASK_WINDOW_SIZE])
                for start in range(0, len(rows), VLLM_TASK_WINDOW_SIZE)
            ]
            window_total = len(batches)
            progress.set_postfix(generated=0, correct=0, windows=f"0/{window_total}")
            next_batch = 0
            busy = [False] * args.data_parallel_size
            stopped = set()
            stop_dispatch = False
            try:
                while len(stopped) < args.data_parallel_size:
                    try:
                        message = result_queue.get(timeout=5)
                    except Exception:
                        failed = [worker.exitcode for worker in workers if worker.exitcode not in (None, 0)]
                        if failed:
                            raise RuntimeError(f"vLLM data-parallel worker exited: {failed}")
                        continue
                    kind, rank, batch_id, payload = message
                    if kind == "error":
                        raise RuntimeError(f"vLLM worker {rank} failed on batch {batch_id}: {payload}")
                    if kind == "stopped":
                        stopped.add(rank)
                        continue
                    if kind == "ready":
                        pass
                    elif kind == "done":
                        busy[rank] = False
                        completed_windows += 1
                        progress.set_postfix(
                            generated=generated_count,
                            correct=saved_count,
                            windows=f"{completed_windows}/{window_total}",
                        )
                        start = batch_id
                        batch_rows = rows[start:start + VLLM_TASK_WINDOW_SIZE]
                        for row, candidate_texts in zip(batch_rows, payload):
                            process_candidates(row, candidate_texts)
                        if args.target_correct_trajectories and saved_count >= args.target_correct_trajectories:
                            stop_dispatch = True
                    else:
                        raise RuntimeError(f"Unexpected vLLM worker message: {kind}")

                    if not busy[rank]:
                        if not stop_dispatch and next_batch < len(batches):
                            batch_start, batch_prompts = batches[next_batch]
                            task_queues[rank].put((batch_start, batch_prompts))
                            busy[rank] = True
                            next_batch += 1
                        else:
                            task_queues[rank].put(None)
                for worker in workers:
                    worker.join()
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
        else:
            for start in range(0, len(rows), TRANSFORMERS_BATCH_SIZE):
                batch_rows = rows[start:start + TRANSFORMERS_BATCH_SIZE]
                batch_prompts = prompts[start:start + TRANSFORMERS_BATCH_SIZE]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        num_beams=num_beams,
                        num_return_sequences=args.num_generations,
                        do_sample=False,
                        early_stopping=True,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                sequence_width = inputs["input_ids"].shape[1]
                for sequence_index, sequence in enumerate(sequences):
                    row = batch_rows[sequence_index // args.num_generations]
                    text = tokenizer.decode(sequence[sequence_width:], skip_special_tokens=True)
                    process_candidates(row, [text])
                    if args.target_correct_trajectories and saved_count >= args.target_correct_trajectories:
                        break
                if args.target_correct_trajectories and saved_count >= args.target_correct_trajectories:
                    break

    print(json.dumps({
        "output": str(args.output), "questions": len(rows), "generations": total,
        "generated_trajectories": generated_count,
        "parsed_final_answers": parsed, "correct_trajectories": saved_count,
        "success_rate": correct / generated_count if generated_count else 0.0,
        "decoding": f"{args.backend}_beam_search", "num_beams": num_beams,
        "data_parallel_size": args.data_parallel_size if args.backend == "vllm" else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
