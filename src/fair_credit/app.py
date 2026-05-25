from __future__ import annotations

from pathlib import Path

import streamlit as st

from fair_credit.metrics import fairness_report
from fair_credit.modeling import load_credit_rows, prepare_demo_predictions


st.set_page_config(page_title="Fair Credit Risk Lab", layout="wide")
st.title("Fair Credit Risk Lab")

default_path = Path(__file__).resolve().parents[2] / "sample_data" / "credit_demo.csv"
threshold = st.slider("Approval threshold", 0.1, 0.9, 0.6, 0.05)

rows = load_credit_rows(default_path)
predictions = prepare_demo_predictions(rows, threshold=threshold)
report = fairness_report(
    predictions["y_true"],
    predictions["y_pred"],
    predictions["y_score"],
    predictions["groups"],
    privileged="A",
    unprivileged="B",
)

st.subheader("Evaluation")
st.json(report)

st.subheader("Sample rows")
st.dataframe(rows)

