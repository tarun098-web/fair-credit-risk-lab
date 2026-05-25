# Architecture

```mermaid
flowchart LR
    A["Credit CSV"] --> B["Preprocessing and scoring"]
    B --> C["Predictions"]
    C --> D["Classification metrics"]
    C --> E["Fairness metrics"]
    C --> F["Calibration metrics"]
    D --> G["Model card outputs"]
    E --> G
    F --> G
    G --> H["Streamlit dashboard"]
```

## Design Notes

- The first version uses synthetic public data so the repository is safe to publish.
- Metric functions are implemented in plain Python to make the fairness logic inspectable.
- The model layer can later be extended with scikit-learn and XGBoost experiments.

