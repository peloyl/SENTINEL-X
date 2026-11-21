# Contributing to SENTINEL-X

Thanks for your interest in contributing!

## Setup
```bash
git clone https://github.com/peloyl/SENTINEL-X.git
cd SENTINEL-X
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/train_model.py
streamlit run app/main.py
```

## Rules of the road
- Run `pytest -q` before opening a PR - all tests must pass (CI runs them too).
- Keep modules decoupled: data, features, models, explainability, reasoning, rules, risk, velocity, audit, monitoring.
- Never commit datasets, trained model files, or the SQLite database.
- Never hardcode metrics or predictions into the dashboard; every number must come from real application data or be labeled as a demonstration.
- Keep model-derived, rule-derived, observed, simulated, and unavailable evidence clearly separated in explanations.
- Fit preprocessing on training data only; resample only inside the training pipeline.

## Pull requests
Use the PR template, describe what changed and how you tested it, and keep PRs focused.
