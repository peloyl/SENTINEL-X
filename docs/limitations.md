
# Limitations
- Kaggle dataset has no user identity, device, IP or geo; velocity/behavioral claims are therefore unavailable or clearly-labeled simulations.
- Default demo data is SYNTHETIC (generated in src/data/loader.py) until the real Kaggle CSV is placed in data/raw/.
- Risk weights 0.6/0.2/0.2 are initial configuration, not validated optima.
- SHAP explains the model, not causality or real-world fraud proof.
- Audit log is append-only at application level only - not tamper-proof, not regulatory-certified.
- Analyst decisions are not ground truth; only verified feedback may be used for retraining, and retraining is manual.
- Monitoring PSI detects data drift, not concept drift without labeled history.
