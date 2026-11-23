"""Prepare the exact GSM-MC training sample used by the original ten-round teacher."""

import argparse
import hashlib
import json
import random
from pathlib import Path


SAMPLE_SIZE = 1024
SAMPLE_SEED = 42


def digest(data):
    return hashlib.sha256(data).hexdigest()


def shared_rows(rows):
    if (len(rows) != SAMPLE_SIZE or len({row['id'] for row in rows}) != SAMPLE_SIZE
            or any(row['split'] != 'train' for row in rows)):
        raise ValueError('Expected exactly 1024 unique shared training rows')
    return rows


def prepare_sample(source):
    lines = source.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if len({row['id'] for row in rows}) != len(rows) or any(row['split'] != 'train' for row in rows):
        raise ValueError('Expected a unique GSM-MC training pool')
    return b''.join(lines[i].rstrip(b'\r\n') + b'\n'
                    for i in random.Random(SAMPLE_SEED).sample(range(len(lines)), SAMPLE_SIZE))


def validate_teacher(metadata, source, sample):
    recorded = metadata.get('train_sha256')
    matches = recorded == digest(sample) or (
        recorded == digest(source) and metadata.get('train_sample_seed') == SAMPLE_SEED
        and metadata.get('max_train_examples') == SAMPLE_SIZE
    )
    if metadata.get('status') != 'complete' or metadata.get('examples') != SAMPLE_SIZE or not matches:
        raise ValueError('Existing teacher does not match the shared 1024 training questions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    sample = prepare_sample(source)
    test = [json.loads(line) for line in args.test.read_bytes().splitlines()]
    test_ids = {row['id'] for row in test}
    if (len(test) != 1319 or len(test_ids) != 1319 or any(row['split'] != 'test' for row in test)
            or test_ids.intersection(json.loads(line)['id'] for line in sample.splitlines())):
        raise ValueError('Expected 1319 unique GSM-MC test rows disjoint from training')
    metadata = args.teacher / 'experiment.json'
    if metadata.is_file():
        validate_teacher(json.loads(metadata.read_text()), source, sample)
    if args.output.exists() and args.output.read_bytes() != sample:
        raise ValueError(f'Existing file does not match the shared sample: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_bytes(sample)
    print(f'Shared training data: {args.output} (1024 rows, SHA256 {digest(sample)})')


if __name__ == '__main__':
    main()
