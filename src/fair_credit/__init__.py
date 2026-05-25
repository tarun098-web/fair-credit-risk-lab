"""Fairness-aware credit risk modelling utilities."""

from .metrics import classification_metrics, equal_opportunity_difference, statistical_parity_difference

__all__ = ["classification_metrics", "equal_opportunity_difference", "statistical_parity_difference"]

