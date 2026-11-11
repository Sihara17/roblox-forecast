"""Historical rank-transition model used by the entrypoint."""

import math
from datetime import datetime


def _hour(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() / 3600.0


def _multiply(left, right):
    size = len(left)
    out = [[0.0] * size for _ in range(size)]
    for i, row in enumerate(left):
        for k, value in enumerate(row):
            if value < 1e-10:
                continue
            for j, next_value in enumerate(right[k]):
                if next_value >= 1e-10:
                    out[i][j] += value * next_value
    return out


def _transition_powers(snapshots):
    size = 101
    counts = [[0.0] * size for _ in range(size)]
    times = [_hour(item['at']) for item in snapshots]
    for i, source in enumerate(snapshots[:-1]):
        candidates = [j for j in range(i + 1, len(snapshots))
                      if 3.0 <= times[j] - times[i] <= 12.0]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(times[k] - times[i] - 6.5))
        destination = {row['universe_id']: row['rank']
                       for row in snapshots[j]['up_and_coming']}
        for row in source['up_and_coming']:
            a = min(size, max(1, int(row['rank']))) - 1
            b = min(size, max(1, int(destination.get(row['universe_id'], 101)))) - 1
            counts[a][b] += 1.0

    transition = []
    for source_rank in range(size):
        row = [0.0] * size
        total = 0.0
        for observed_rank in range(size):
            proximity = math.exp(-0.5 * ((observed_rank - source_rank) / 2.4) ** 2)
            if proximity < 0.008:
                continue
            for target_rank, count in enumerate(counts[observed_rank]):
                row[target_rank] += proximity * count
                total += proximity * count
        for target_rank in range(size - 1):
            prior = 0.12 * math.exp(-0.5 * ((target_rank - source_rank) / 2.2) ** 2)
            row[target_rank] += prior
            total += prior
        row[-1] += 0.08
        total += 0.08
        transition.append([value / total for value in row])
    transition[-1] = [0.0] * size
    transition[-1][-1] = 1.0
    powers = [[[1.0 if i == j else 0.0 for j in range(size)]
               for i in range(size)]]
    for _ in range(12):
        powers.append(_multiply(powers[-1], transition))
    return powers


def _rank_slopes(snapshots):
    latest_time = _hour(snapshots[-1]['at'])
    latest = snapshots[-1]['up_and_coming']
    histories = {row['universe_id']: [] for row in latest}
    for snapshot in snapshots:
        t = _hour(snapshot['at'])
        if latest_time - t > 72.0:
            continue
        for row in snapshot['up_and_coming']:
            uid = row['universe_id']
            if uid in histories:
                histories[uid].append((t, float(row['rank'])))
    slopes = {}
    for uid, points in histories.items():
        points = points[-6:]
        if len(points) < 2:
            slopes[uid] = 0.0
            continue
        origin = points[-1][0]
        xs = [t - origin for t, _ in points]
        ys = [rank for _, rank in points]
        weights = [math.exp(x / 36.0) for x in xs]
        weight_sum = sum(weights)
        mean_x = sum(w * x for w, x in zip(weights, xs)) / weight_sum
        mean_y = sum(w * y for w, y in zip(weights, ys)) / weight_sum
        variance = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
        slope = (sum(w * (x - mean_x) * (y - mean_y)
                     for w, x, y in zip(weights, xs, ys)) / variance
                 if variance > 1e-9 else 0.0)
        slopes[uid] = max(-0.12, min(0.12, slope)) * 0.35
    return slopes


def predict(snapshots, places):
    current = sorted(snapshots[-1]['up_and_coming'], key=lambda row: row['rank'])
    ids = [row['universe_id'] for row in current]
    ranks = {row['universe_id']: int(row['rank']) for row in current}
    slopes = _rank_slopes(snapshots)
    powers = _transition_powers(snapshots)
    result = {}
    for uid in ids:
        for place in range(1, places + 1):
            probability = 0.0
            for hour in range(3, 73, 3):
                effective = max(1.0, min(100.0, ranks[uid] + slopes[uid] * hour))
                low = int(math.floor(effective))
                high = min(100, low + 1)
                fraction = effective - low
                step = min(12, max(0, int(round(hour / 6.5))))
                matrix = powers[step]
                probability += ((1.0 - fraction) * matrix[low - 1][place - 1]
                                + fraction * matrix[high - 1][place - 1])
            result[(uid, place)] = probability / 24.0
    return ids, result
