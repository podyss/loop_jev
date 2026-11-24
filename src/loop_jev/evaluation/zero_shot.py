#!/usr/bin/env python3
"""Score answer options with vLLM BF16 inference and FP32 probabilities by default."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_rows(path, limit=None):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def render_prompt(tokenizer, row, latent_slots=0, latent_marker_token_id=None, *, handoff=False):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"({labels[i]}) {option}" for i, option in enumerate(row["options"]))
    text = f"Question: {row['question']}\nOptions:\n{options}"
    if handoff:
        from loop_jev.staged_handoff import encode_handoff_prompt
        prompt_ids, option_start = encode_handoff_prompt(tokenizer, row['question'], text)
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    answer_slot = tokenizer.encode("Answer: (", add_special_tokens=False)
    if latent_slots:
        if latent_marker_token_id is None:
            raise ValueError("Latent checkpoint is missing its marker token ID")
        prompt_ids += [latent_marker_token_id] * latent_slots
    prompt_ids = prompt_ids + answer_slot
    return (prompt_ids, option_start) if handoff else prompt_ids


def score_vllm(args, rows, tokenizer, candidate_ids):
    from vllm import LLM, SamplingParams

    prompts = [{"prompt_token_ids": render_prompt(tokenizer, row, args._latent_slots,
                                                  args._latent_marker_token_id)} for row in rows]
    model = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        generation_config="vllm",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_model_len=max(len(prompt["prompt_token_ids"]) for prompt in prompts) + 1,
        max_logprobs=len(candidate_ids),
        # Return probabilities after masking to valid answer labels, as in the
        # existing evaluator. Raw top-k logprobs may omit low-scoring labels.
        logprobs_mode="processed_logprobs",
    )
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            logprobs=len(row["options"]),
            allowed_token_ids=candidate_ids[:len(row["options"])],
            detokenize=False,
        )
        for row in rows
    ]
    # Submit the whole local shard; vLLM schedules batches using its own limits
    # and available KV cache rather than a fixed application batch size.
    outputs = model.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(rows):
        raise RuntimeError("vLLM returned an unexpected number of results")
    for row, output in zip(rows, outputs):
        completion = output.outputs[0]
        if len(completion.token_ids) != 1 or not completion.logprobs:
            raise RuntimeError(f"Missing one-token answer logprobs for {row['id']}")
        logprobs = completion.logprobs[0]
        valid_ids = candidate_ids[:len(row["options"])]
        if any(token_id not in logprobs for token_id in valid_ids):
            raise RuntimeError(f"Missing answer-label logprobs for {row['id']}")
        yield row, torch.tensor(
            [logprobs[token_id].logprob for token_id in valid_ids], dtype=torch.float32,
        )


def score_transformers(args, rows, tokenizer, candidate_ids):
    dtype = getattr(torch, args.dtype)
    from loop_jev.transition import load_loop_model
    model = load_loop_model(
        args.model, dtype=dtype, device_map={"": 0},
        attn_implementation="sdpa", trust_remote_code=False,
    ).eval()
    option_workspace = getattr(model.config, "option_workspace", False)
    direct_answer_head = getattr(model.config, "direct_answer_head", False)
    shared_answer_head = direct_answer_head == "shared"
    refinement_ids = getattr(model.config, "loop_refinement_token_ids", None)
    refinement = bool(refinement_ids) and not args.disable_looping
    if refinement and refinement_ids != candidate_ids:
        raise ValueError("Refinement checkpoint label tokens do not match the evaluation tokenizer")
    handoff = getattr(model.config, "loop_staged_handoff", False) and not args.disable_looping
    if option_workspace:
        from loop_jev.option_workspace import encode_option_prompt, pad_option_spans
    if shared_answer_head:
        from loop_jev.option_workspace import encode_choice_prompt, pad_option_end_positions
    if not args.disable_looping or option_workspace:
        from loop_jev.looping import enable_configured_looping
        enable_configured_looping(model, use_transition=not args.disable_transition,
                                  readout_round=args.loop_readout_round, bypass_looping=args.disable_looping,
                                  reset_latent_core=args.reset_latent_core)
    device = model.get_input_embeddings().weight.device
    rank = args._worker_rank
    desc = "Evaluating" if rank is None else f"Evaluating rank {rank}"
    with tqdm(total=len(rows), desc=desc, unit="example", dynamic_ncols=True,
              position=rank or 0) as progress:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            if option_workspace or shared_answer_head:
                encode_choices = encode_choice_prompt if shared_answer_head else encode_option_prompt
                encoded = [encode_choices(tokenizer, row) for row in batch_rows]
                suffix = tokenizer.encode("Answer: (", add_special_tokens=False)
                prompts = [ids + [args._latent_marker_token_id] * args._latent_slots + suffix for ids, _ in encoded]
            elif handoff:
                encoded = [render_prompt(tokenizer, row, args._latent_slots, args._latent_marker_token_id,
                                         handoff=True) for row in batch_rows]
                prompts, handoff_offsets = zip(*encoded)
            else:
                prompts = [render_prompt(tokenizer, row, args._latent_slots,
                                         args._latent_marker_token_id) for row in batch_rows]
            max_length = max(map(len, prompts))
            option_kwargs = {}
            if args.disable_evidence_reread:
                option_kwargs["disable_evidence_reread"] = True
            if refinement:
                option_kwargs["refine_answer_scores"] = True
            if direct_answer_head:
                option_kwargs["direct_answer_scores"] = True
            if shared_answer_head:
                option_kwargs["direct_answer_positions"] = pad_option_end_positions(
                    [spans for _, spans in encoded], [max_length - len(ids) for ids in prompts],
                ).to(device)
            if option_workspace:
                option_kwargs["option_spans"] = pad_option_spans(
                    [spans for _, spans in encoded], [max_length - len(ids) for ids in prompts],
                ).to(device)
            if handoff:
                option_kwargs["handoff_option_start"] = torch.tensor([
                    offset + max_length - len(prompt) for offset, prompt in zip(handoff_offsets, prompts)
                ], device=device)
                option_kwargs["swap_handoff"] = args.swap_handoff
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            input_ids = torch.full((len(prompts), max_length), pad_id, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, prompt in enumerate(prompts):
                offset = max_length - len(prompt)
                input_ids[index, offset:] = torch.tensor(prompt, dtype=torch.long)
                attention_mask[index, offset:] = 1
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    logits_to_keep=1,
                    use_cache=False,
                    **option_kwargs,
                ).logits
            scores = (logits if option_workspace or refinement or direct_answer_head
                      else logits[:, -1, candidate_ids]).float().cpu()
            for row, row_scores in zip(batch_rows, scores):
                yield row, row_scores[:len(row["options"])]
            progress.update(len(batch_rows))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Transformers backend only; vLLM schedules batches automatically")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm",
                        help="Inference engine")
    parser.add_argument("--disable-looping", action="store_true",
                        help="Run a loop-configured checkpoint with one pass through each layer")
    parser.add_argument("--disable-latent-slots", action="store_true",
                        help="Remove workspace markers as well as looping for the direct ablation")
    parser.add_argument("--loop-readout-round", type=int,
                        help="Stop at this round while retaining the training-time damping step size")
    parser.add_argument("--reset-latent-core", action="store_true",
                        help="Reset the dedicated latent state just before the final round")
    parser.add_argument("--disable-transition", action="store_true",
                        help="Bypass the transition head while retaining checkpoint looping and damping")
    parser.add_argument("--disable-evidence-reread", action="store_true",
                        help="Bypass both evidence reads while retaining checkpoint looping and damping")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Inference weight/compute dtype; probabilities use FP32")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="vLLM per-GPU memory fraction (verl rollout default: 0.5)")
    parser.add_argument("--swap-handoff", action="store_true",
                        help="Rotate first-round latent states across examples in each evaluation batch")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable vLLM compilation/CUDA graphs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()
    if args.disable_latent_slots and not args.disable_looping:
        parser.error("--disable-latent-slots requires --disable-looping")
    if args.loop_readout_round is not None and (args.loop_readout_round < 1
                                                or args.disable_looping or args.backend != "transformers"):
        parser.error("--loop-readout-round requires a positive round and enabled Transformers looping")
    if args.disable_transition and (args.disable_looping or args.backend != "transformers"):
        parser.error("--disable-transition requires --backend transformers and enabled looping")
    if args.disable_evidence_reread and (args.disable_looping or args.backend != "transformers"
                                         or args.loop_readout_round is not None):
        parser.error("--disable-evidence-reread requires Transformers at the full loop depth")
    if args.swap_handoff and (args.disable_looping or args.backend != "transformers" or args.batch_size < 2):
        parser.error("--swap-handoff requires enabled Transformers looping and batches of at least two")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    rows = read_rows(args.input, args.limit)
    if not rows:
        raise ValueError("Input dataset is empty")
    if args.data_parallel_size < 1:
        raise ValueError("--data-parallel-size must be positive")
    if args._worker_rank is None and args.data_parallel_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = ([item.strip() for item in visible.split(",") if item.strip()]
                   if visible is not None else [str(i) for i in range(torch.cuda.device_count())])
        if len(gpu_ids) < args.data_parallel_size:
            raise ValueError(f"Requested {args.data_parallel_size} DP workers, found GPUs {gpu_ids}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jev-dp-") as tmp:
            shard_paths = [Path(tmp) / f"rank-{rank}.jsonl" for rank in range(args.data_parallel_size)]
            processes = []
            for rank, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--model", args.model, "--input", str(args.input),
                    "--output", str(shard_path), "--batch-size", str(args.batch_size),
                    "--dtype", args.dtype, "--data-parallel-size", str(args.data_parallel_size),
                    "--backend", args.backend, "--seed", str(args.seed),
                    "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                    "--_worker-rank", str(rank),
                ]
                if args.enforce_eager:
                    command.append("--enforce-eager")
                if args.disable_looping:
                    command.append("--disable-looping")
                if args.disable_latent_slots:
                    command.append("--disable-latent-slots")
                if args.reset_latent_core:
                    command.append("--reset-latent-core")
                if args.disable_transition:
                    command.append("--disable-transition")
                if args.disable_evidence_reread:
                    command.append("--disable-evidence-reread")
                if args.swap_handoff:
                    command.append("--swap-handoff")
                if args.loop_readout_round is not None:
                    command.extend(["--loop-readout-round", str(args.loop_readout_round)])
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])
                worker_env = dict(os.environ)
                worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                worker_env["HF_ENDPOINT"] = "https://hf-mirror.com"
                processes.append(subprocess.Popen(command, env=worker_env))
            statuses = [process.wait() for process in processes]
            failures = [status for status in statuses if status != 0]
            if failures:
                raise subprocess.CalledProcessError(failures[0], "DP evaluation worker")

            by_id = {}
            for shard_path in shard_paths:
                for row in read_rows(shard_path):
                    by_id[row["id"]] = row
            if len(by_id) != len(rows):
                raise RuntimeError(f"DP merge mismatch: expected {len(rows)} rows, got {len(by_id)}")
            results = [by_id[row["id"]] for row in rows]
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
            encoding="utf-8",
        )
        groups = defaultdict(lambda: [0, 0])
        for row in results:
            group = f"{row.get('dataset', 'unknown')}/{row.get('split', 'unknown')}"
            groups[group][0] += int(row["correct_prediction"])
            groups[group][1] += 1
        summary = {name: {"correct": correct, "total": total, "accuracy": correct / total}
                   for name, (correct, total) in groups.items()}
        print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
        return

    if args._worker_rank is not None:
        rows = rows[args._worker_rank::args.data_parallel_size]
    if not rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        return
    for row in rows:
        if not 1 <= len(row["options"]) <= 26:
            raise ValueError(f"Expected 1–26 answer options for {row['id']}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if args.disable_evidence_reread and not getattr(config, "loop_evidence_reread", False):
        parser.error("--disable-evidence-reread requires an evidence reread checkpoint")
    if args.swap_handoff and (not getattr(config, "loop_staged_handoff", False)
                              or len(rows) < 2 or len(rows) % args.batch_size == 1):
        parser.error("--swap-handoff requires a handoff checkpoint and no singleton batches on each shard")
    if getattr(config, "loop_staged_handoff", False) and args.loop_readout_round == 1:
        parser.error("Staged handoff requires at least two rounds; use --disable-looping for a standard single pass")
    if getattr(config, "option_workspace", False) and (
        args.backend != "transformers" or any(len(row["options"]) != 4 for row in rows)
    ):
        parser.error("Option workspace checkpoints require --backend transformers and four choices per row")
    direct_answer_head = getattr(config, "direct_answer_head", False)
    if direct_answer_head and args.backend != "transformers":
        parser.error("Direct answer head checkpoints require --backend transformers")
    if direct_answer_head is True and any(len(row["options"]) != 4 for row in rows):
        parser.error("This fixed-width checkpoint requires four choices per row")
    if args.loop_readout_round is not None and (getattr(config, "loop_count", 1) < 2
                                                or args.loop_readout_round > config.loop_count):
        parser.error("--loop-readout-round requires a looped checkpoint and cannot exceed its trained rounds")
    if args.reset_latent_core and (not getattr(config, "latent_core_width", 0) or args.disable_looping
                                   or args.loop_readout_round is not None or args.backend != "transformers"):
        parser.error("--reset-latent-core requires the full dedicated core and --backend transformers")
    if args.disable_latent_slots and getattr(config, "option_workspace", False):
        parser.error("Option workspace checkpoints do not support --disable-latent-slots")
    args._latent_slots = 0 if args.disable_latent_slots else getattr(config, "latent_slots", 0)
    args._latent_marker_token_id = getattr(config, "latent_marker_token_id", None)
    if args._latent_slots and (args._latent_marker_token_id is None
                               or not 0 <= args._latent_marker_token_id < len(tokenizer)):
        parser.error("Latent checkpoint has an invalid marker token ID")
    if (getattr(config, "loop_count", 1) > 1 and not args.disable_looping
            and args.backend == "vllm"):
        parser.error("Looped checkpoints require --backend transformers")
    if getattr(config, "loop_transition_rank", 0) > 0 and args.backend != "transformers":
        parser.error("Transition checkpoints require --backend transformers, including --disable-looping")
    if getattr(config, "loop_refinement_token_ids", None) and args.loop_readout_round == 1:
        parser.error("Use --disable-looping for the undamped first-pass refinement readout")
    if args.disable_transition and (getattr(config, "loop_transition_rank", 0) <= 0
                                    or getattr(config, "loop_count", 1) <= 1):
        parser.error("--disable-transition requires a checkpoint with a transition head and multiple loops")
    candidate_ids = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate label {label!r} is not a single token: {ids}")
        candidate_ids.append(ids[0])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    torch.cuda.set_device(0)
    score = score_vllm if args.backend == "vllm" else score_transformers
    results = []
    for row, scores in score(args, rows, tokenizer, candidate_ids):
        probs = torch.softmax(scores, dim=-1).tolist()
        pred = int(torch.argmax(scores).item())
        results.append({
            **row,
            "model": args.model,
            "backend": args.backend,
            **({"readout": "direct-answer-head"} if direct_answer_head else {}),
            "dtype": args.dtype,
            "disable_looping": args.disable_looping,
            **({"latent_slots": 0} if args.disable_latent_slots else {}),
            "disable_transition": args.disable_transition,
            **({"swap_handoff": args.swap_handoff} if getattr(config, "loop_staged_handoff", False) else {}),
            **({"disable_evidence_reread": args.disable_evidence_reread}
               if getattr(config, "loop_evidence_reread", False) else {}),
            "loop_readout_round": args.loop_readout_round,
            "reset_latent_core": args.reset_latent_core,
            "probability_dtype": "float32",
            "prediction_index": pred,
            "prediction_label": chr(ord("A") + pred),
            "correct_prediction": pred == row["gold_index"],
            "option_probabilities": probs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups = defaultdict(lambda: [0, 0])
    for row in results:
        group = f"{row.get('dataset', 'unknown')}/{row.get('split', 'unknown')}"
        groups[group][0] += int(row["correct_prediction"])
        groups[group][1] += 1
    summary = {name: {"correct": correct, "total": total, "accuracy": correct / total}
               for name, (correct, total) in groups.items()}
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
