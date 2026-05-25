from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    tp = fp = tn = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    precision = safe_divide(counts.tp, counts.tp + counts.fp)
    recall = safe_divide(counts.tp, counts.tp + counts.fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(counts.tp + counts.tn, len(y_true))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def group_values(groups: Sequence[str]) -> list[str]:
    return sorted(set(groups))


def positive_rate(y_pred: Sequence[int], groups: Sequence[str], group: str) -> float:
    values = [pred for pred, value in zip(y_pred, groups) if value == group]
    return safe_divide(sum(values), len(values))


def true_positive_rate(y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str], group: str) -> float:
    positives = [(actual, pred) for actual, pred, value in zip(y_true, y_pred, groups) if value == group and actual == 1]
    return safe_divide(sum(pred for _, pred in positives), len(positives))


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str], group: str) -> float:
    negatives = [(actual, pred) for actual, pred, value in zip(y_true, y_pred, groups) if value == group and actual == 0]
    return safe_divide(sum(pred for _, pred in negatives), len(negatives))


def statistical_parity_difference(y_pred: Sequence[int], groups: Sequence[str], privileged: str, unprivileged: str) -> float:
    return round(positive_rate(y_pred, groups, unprivileged) - positive_rate(y_pred, groups, privileged), 4)


def demographic_parity_difference(y_pred: Sequence[int], groups: Sequence[str]) -> float:
    rates = [positive_rate(y_pred, groups, group) for group in group_values(groups)]
    return round(max(rates) - min(rates), 4) if rates else 0.0


def equal_opportunity_difference(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    privileged: str,
    unprivileged: str,
) -> float:
    return round(
        true_positive_rate(y_true, y_pred, groups, unprivileged)
        - true_positive_rate(y_true, y_pred, groups, privileged),
        4,
    )


def equalized_odds_difference(y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str]) -> float:
    values = []
    for group in group_values(groups):
        values.append(true_positive_rate(y_true, y_pred, groups, group))
        values.append(false_positive_rate(y_true, y_pred, groups, group))
    return round(max(values) - min(values), 4) if values else 0.0


def calibration_bins(y_true: Sequence[int], y_score: Sequence[float], bins: int = 5) -> list[dict[str, float]]:
    output = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (actual, score)
            for actual, score in zip(y_true, y_score)
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        if not bucket:
            continue
        actual_rate = safe_divide(sum(actual for actual, _ in bucket), len(bucket))
        mean_score = safe_divide(sum(score for _, score in bucket), len(bucket))
        output.append(
            {
                "low": low,
                "high": high,
                "count": len(bucket),
                "mean_score": round(mean_score, 4),
                "actual_rate": round(actual_rate, 4),
                "absolute_error": round(abs(mean_score - actual_rate), 4),
            }
        )
    return output


def mean_absolute_calibration_error(y_true: Sequence[int], y_score: Sequence[float], bins: int = 5) -> float:
    bucket_stats = calibration_bins(y_true, y_score, bins=bins)
    total = sum(bucket["count"] for bucket in bucket_stats)
    if total == 0:
        return 0.0
    weighted_error = sum(bucket["absolute_error"] * bucket["count"] for bucket in bucket_stats)
    return round(weighted_error / total, 4)


def fairness_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    groups: Sequence[str],
    privileged: str,
    unprivileged: str,
) -> dict[str, float]:
    return {
        **classification_metrics(y_true, y_pred),
        "statistical_parity_difference": statistical_parity_difference(y_pred, groups, privileged, unprivileged),
        "demographic_parity_difference": demographic_parity_difference(y_pred, groups),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, groups, privileged, unprivileged),
        "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, groups),
        "mace": mean_absolute_calibration_error(y_true, y_score),
    }

