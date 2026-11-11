#!/usr/bin/env python3
"""Read chart history and write its place-by-place forecast."""

import json
import os
from pathlib import Path

from forecast_model import predict


def main():
    input_dir = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    output_dir = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    snapshots = [json.loads(line) for line in
                 (input_dir / 'history.ndjson').read_text(encoding='utf-8').splitlines()
                 if line.strip()]
    task = json.loads((input_dir / 'task.json').read_text(encoding='utf-8'))
    places = int(task['places'])
    ids, rank_probability = predict(snapshots, places)

    probabilities = []
    for place in range(1, places + 1):
        candidates = sorted(ids, key=lambda uid: rank_probability[(uid, place)],
                            reverse=True)[:60]
        mass = sum(rank_probability[(uid, place)] for uid in candidates)
        scale = min(1.0, 0.94 / mass) if mass > 0.0 else 1.0
        for uid in candidates:
            value = rank_probability[(uid, place)] * scale
            if value > 1e-8:
                probabilities.append({
                    'universe_id': uid,
                    'place': place,
                    'p': round(value, 8),
                })

    remaining = set(ids)
    top10 = []
    for place in range(1, places + 1):
        chosen = max(remaining, key=lambda uid: rank_probability[(uid, place)])
        top10.append(chosen)
        remaining.remove(chosen)

    output_dir.mkdir(parents=True, exist_ok=True)
    answer = {'probabilities': probabilities, 'top10': top10}
    (output_dir / task['output']).write_text(
        json.dumps(answer, separators=(',', ':')) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
