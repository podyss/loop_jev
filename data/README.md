# Data

`jev-mathqa/` contains normalized MathQA train/dev/validation/test splits used by the JEV experiments. `compmath-mcq/test.jsonl` is an external multiple-choice evaluation set. Dataset provenance, split construction, row counts, and SHA-256 checksums are recorded in `jev-datasets-manifest.json`.

The prepared files are bundled so experiments can run without an implicit dependency on the parent workspace. To regenerate them from Hugging Face, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run python -m loop_jev.data.prepare --output-dir data
```

Review the upstream dataset cards and licenses before redistributing these datasets. The prepared MathQA and CompMath-MCQ files are included for research reproducibility; this project does not grant additional rights to their contents.
