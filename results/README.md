# Archived Results

This directory contains experiment metadata and prediction JSONL files copied from the parent workspace. Prediction records retain the original row-level scores and identifiers. Each `experiment.json` captures training configuration and dataset checksum.

Large model checkpoints are not copied here. In the source workspace they remain under `outputs/`; the original checkpoint folder names are preserved in the experiment records where available. Some historical runs used local model paths that will need to be replaced when reproducing on another machine.
