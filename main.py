#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart from its observed history."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() / 3600.0


def normal_kernel(place: int, observed_rank: int, width: float) -> float:
    """A normalized, discretized Gaussian around an observed future rank."""
    low = max(1, int(observed_rank - 4.5 * width))
    high = min(100, int(observed_rank + 4.5 * width) + 1)
    denominator = 0.0
    numerator = 0.0
    for rank in range(low, high + 1):
        weight = math.exp(-0.5 * ((rank - observed_rank) / width) ** 2)
        denominator += weight
        if rank == place:
            numerator = weight
    return numerator / denominator if denominator else 0.0


def fit_slopes(snapshots: list[dict[str, Any]], times: list[float]) -> dict[int, float]:
    """Estimate recent, shrunk rank momentum for games in the latest chart."""
    latest = snapshots[-1]["up_and_coming"]
    latest_ids = {row["universe_id"] for row in latest}
    histories: dict[int, list[tuple[float, int]]] = {uid: [] for uid in latest_ids}
    cutoff = times[-1] - 72.0
    for t, snap in zip(times, snapshots):
        if t < cutoff:
            continue
        for row in snap["up_and_coming"]:
            uid = row["universe_id"]
            if uid in histories:
                histories[uid].append((t, row["rank"]))

    slopes: dict[int, float] = {}
    for uid, points in histories.items():
        if len(points) < 2:
            slopes[uid] = 0.0
            continue
        # Recent observations carry more weight; this reduces the effect of
        # old chart regimes while retaining a stable trend estimate.
        recent = points[-6:]
        origin = recent[-1][0]
        xs = [t - origin for t, _ in recent]
        ys = [float(rank) for _, rank in recent]
        weights = [math.exp(x / 36.0) for x in xs]
        sw = sum(weights)
        mx = sum(w * x for w, x in zip(weights, xs)) / sw
        my = sum(w * y for w, y in zip(weights, ys)) / sw
        variance = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
        slope = (sum(w * (x - mx) * (y - my) for w, x, y in zip(weights, xs, ys)) / variance
                 if variance > 1e-9 else 0.0)
        # Rank changes are noisy. Shrink towards persistence and cap extreme
        # extrapolations so a brief jump cannot dominate a three-day forecast.
        slopes[uid] = max(-0.12, min(0.12, slope)) * 0.35
    return slopes


def learn_transition_powers(snapshots: list[dict[str, Any]], times: list[float]) -> list[list[list[float]]]:
    """Learn a smoothed six-hour rank transition model and its powers.

    State 101 means a game left the chart. Training pairs are near six hours
    apart, so dense hourly snapshots at the end of the history do not overwhelm
    the more informative multi-hour archive transitions.
    """
    size = 101
    counts = [[0.0] * size for _ in range(size)]
    for i, source in enumerate(snapshots[:-1]):
        possible = [j for j in range(i + 1, len(snapshots))
                    if 3.0 <= times[j] - times[i] <= 12.0]
        if not possible:
            continue
        j = min(possible, key=lambda index: abs((times[index] - times[i]) - 6.5))
        destination = {row["universe_id"]: row["rank"] for row in snapshots[j]["up_and_coming"]}
        for row in source["up_and_coming"]:
            source_rank = min(size, max(1, int(row["rank"]))) - 1
            target_rank = min(size, max(1, int(destination.get(row["universe_id"], 101)))) - 1
            counts[source_rank][target_rank] += 1.0

    transition: list[list[float]] = []
    for source_rank in range(size):
        weighted = [0.0] * size
        total = 0.0
        for observed_rank in range(size):
            proximity = math.exp(-0.5 * ((observed_rank - source_rank) / 2.4) ** 2)
            if proximity < 0.01:
                continue
            for destination_rank, count in enumerate(counts[observed_rank]):
                weighted[destination_rank] += proximity * count
                total += proximity * count
        # A small persistence prior keeps the sparse archive from producing
        # brittle zeros; it decays into adjacent ranks instead of forcing a
        # game's place to stay fixed.
        prior_width = 2.2
        for destination_rank in range(size - 1):
            prior = 0.12 * math.exp(-0.5 * ((destination_rank - source_rank) / prior_width) ** 2)
            weighted[destination_rank] += prior
            total += prior
        weighted[-1] += 0.08
        total += 0.08
        transition.append([value / total for value in weighted])

    # Once off-chart, the same game is treated as gone; new games are accounted
    # for by the unassigned probability mass in the output.
    transition[-1] = [0.0] * size
    transition[-1][-1] = 1.0

    def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
        result = [[0.0] * size for _ in range(size)]
        for i, row in enumerate(left):
            for k, value in enumerate(row):
                if value <= 1e-10:
                    continue
                for j, next_value in enumerate(right[k]):
                    if next_value > 1e-10:
                        result[i][j] += value * next_value
        return result

    powers = [ [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)] ]
    for _ in range(12):
        powers.append(multiply(powers[-1], transition))
    return powers


def forecast() -> dict[str, Any]:
    snapshots = [json.loads(line) for line in (INPUT / "history.ndjson").read_text().splitlines() if line.strip()]
    task = json.loads((INPUT / "task.json").read_text())
    if not snapshots:
        raise ValueError("history.ndjson is empty")

    times = [parse_time(snap["at"]) for snap in snapshots]
    latest_rows = sorted(snapshots[-1]["up_and_coming"], key=lambda row: row["rank"])
    current_ids = [row["universe_id"] for row in latest_rows]
    current_rank = {row["universe_id"]: row["rank"] for row in latest_rows}
    places = int(task["places"])
    slopes = fit_slopes(snapshots, times)
    powers = learn_transition_powers(snapshots, times)

    # The history contains a mix of short recorder intervals and longer public
    # archive gaps. For every forecast hour, borrow transitions at similar
    # elapsed times and starting ranks. The kernel smooths discrete ranks while
    # off-chart outcomes remain in the denominator, preserving attrition.
    steps = range(3, 73, 3)
    rank_prob: dict[tuple[int, int], float] = {}
    for uid in current_ids:
        start = current_rank[uid]
        for place in range(1, places + 1):
            total = 0.0
            for horizon in steps:
                effective_start = max(1.0, min(100.0, start + slopes.get(uid, 0.0) * horizon))
                lower = int(math.floor(effective_start))
                upper = min(100, lower + 1)
                fraction = effective_start - lower
                transitions_ahead = min(12, max(0, int(round(horizon / 6.5))))
                low_p = powers[transitions_ahead][lower - 1][place - 1]
                high_p = powers[transitions_ahead][upper - 1][place - 1]
                total += low_p * (1.0 - fraction) + high_p * fraction
            rank_prob[(uid, place)] = total / len(steps)

    # Keep the file compact and retain probability mass for as-yet unseen
    # games. Unnamed mass is shared by the scorer across chart games omitted
    # from this place's rows.
    probabilities: list[dict[str, Any]] = []
    for place in range(1, places + 1):
        ordered = sorted(current_ids, key=lambda uid: rank_prob[(uid, place)], reverse=True)[:60]
        mass = sum(rank_prob[(uid, place)] for uid in ordered)
        scale = min(1.0, 0.94 / mass) if mass > 0 else 1.0
        for uid in ordered:
            p = rank_prob[(uid, place)] * scale
            if p > 1e-8:
                probabilities.append({"universe_id": uid, "place": place, "p": round(p, 8)})

    # The tie-break's ordered top 10 is derived from the same horizon-averaged
    # marginals, with a unique game at each place.
    remaining = set(current_ids)
    top10: list[int] = []
    for place in range(1, places + 1):
        uid = max(remaining, key=lambda candidate: rank_prob[(candidate, place)])
        top10.append(uid)
        remaining.remove(uid)
    return {"probabilities": probabilities, "top10": top10}


def main() -> None:
    result = forecast()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "forecast.json"
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
