
# SENTINEL-X Architecture
```text
Data (Kaggle or labeled SYNTHETIC demo) -> Loader/Validation -> Features (fit on train only)
 -> Logistic Regression (baseline) + XGBoost (primary) -> SHAP Explainer
 -> YAML Rule Engine + Velocity Engine (available/simulated/unavailable)
 -> Risk Aggregation (0.6/0.2/0.2, configurable) -> Reasoning Engine (deterministic templates)
 -> Streamlit Dashboard (5 pages) -> SQLite Audit/Decisions/Feedback -> Monitoring (PSI drift)
```
Modules communicate via plain dicts/DataFrames. No external APIs. Model, rule and reasoning evidence are kept separate.
