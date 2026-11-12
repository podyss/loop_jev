# Environment

## Verified machine

The source workspace was checked on 2026-11-12 with:

- Ubuntu/Linux, Python 3.12.14
- 4 x NVIDIA GeForce RTX 3090, driver 580.126.09, 24 GiB per GPU
- PyTorch 2.11.0+cu130
- Installed: Transformers 5.21.0, Accelerate 1.11.0, Datasets 3.6.0, PyArrow 26.0.0, vLLM 0.26.0.

The dependency declaration uses Transformers 5.21.0, matching the inspected parent virtual environment and the tensor-parallel `DistributedConfig` API used by answer-slot distillation. The declared PEFT 0.18.1 differs from the installed 0.17.1 in that environment. CUDA is supplied by the PyTorch wheel; the host still needs an NVIDIA driver compatible with that wheel. The four-GPU machine is a record of the source environment, not a minimum hardware requirement. Full-parameter Qwen3-1.7B training uses multiple GPUs in the supplied commands; lower-memory configurations may need smaller batches or a different sharding strategy.

## Create an environment

Install `uv`, then from this directory:

```bash
uv sync --python 3.12
```

Set the Hugging Face mirror before downloading models or datasets:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Verify the environment:

```bash
uv run python - <<'PY'
import torch, transformers, datasets
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("GPU count", torch.cuda.device_count())
PY
```

## Compatibility note

The JEV code was developed in the source workspace with its installed Transformers 5.21.0, including the distributed tensor-parallel API used by `src/loop_jev/training/answer_slot_distill.py`. The migrated project declaration pins 5.5.3 as requested from the parent `pyproject.toml`; run the CLI/import checks after environment creation before launching a training job. Record actual package versions in new experiment metadata.
