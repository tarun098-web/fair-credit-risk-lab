from __future__ import annotations

from pathlib import Path
import csv


def load_credit_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rule_based_credit_score(row: dict[str, str]) -> float:
    income = float(row["income"])
    debt = float(row["debt"])
    late_payments = int(row["late_payments"])
    employment_years = float(row["employment_years"])

    score = 0.55
    score += min(income / 120000, 0.25)
    score -= min(debt / 80000, 0.25)
    score -= late_payments * 0.08
    score += min(employment_years / 20, 0.15)
    return max(0.0, min(1.0, round(score, 4)))


def predict_from_scores(scores: list[float], threshold: float = 0.6) -> list[int]:
    return [1 if score >= threshold else 0 for score in scores]


def prepare_demo_predictions(rows: list[dict[str, str]], threshold: float = 0.6) -> dict[str, list]:
    scores = [rule_based_credit_score(row) for row in rows]
    return {
        "y_true": [int(row["approved"]) for row in rows],
        "y_score": scores,
        "y_pred": predict_from_scores(scores, threshold=threshold),
        "groups": [row["group"] for row in rows],
    }

