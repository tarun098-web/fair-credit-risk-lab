from fair_credit.metrics import (
    calibration_bins,
    classification_metrics,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    mean_absolute_calibration_error,
    statistical_parity_difference,
)
from fair_credit.modeling import predict_from_scores, prepare_demo_predictions, rule_based_credit_score


def test_classification_metrics():
    metrics = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_statistical_parity_difference():
    value = statistical_parity_difference([1, 1, 0, 0], ["A", "B", "A", "B"], "A", "B")
    assert value == 0.0


def test_demographic_parity_difference():
    value = demographic_parity_difference([1, 1, 0, 0], ["A", "B", "A", "B"])
    assert value == 0.0


def test_equal_opportunity_difference():
    value = equal_opportunity_difference([1, 1, 1, 0], [1, 0, 1, 0], ["A", "B", "B", "A"], "A", "B")
    assert value == -0.5


def test_equalized_odds_difference():
    value = equalized_odds_difference([1, 1, 0, 0], [1, 0, 1, 0], ["A", "B", "A", "B"])
    assert value == 1.0


def test_calibration_error():
    bins = calibration_bins([0, 1], [0.2, 0.8], bins=2)
    assert len(bins) == 2
    assert mean_absolute_calibration_error([0, 1], [0.2, 0.8], bins=2) == 0.2


def test_rule_based_predictions():
    rows = [
        {"income": "80000", "debt": "10000", "late_payments": "0", "employment_years": "8", "approved": "1", "group": "A"}
    ]
    output = prepare_demo_predictions(rows)
    assert rule_based_credit_score(rows[0]) > 0.6
    assert predict_from_scores(output["y_score"]) == [1]

