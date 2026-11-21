
"""Data loading & validation. Real Kaggle file if present, else clearly-labeled SYNTHETIC demo."""
from pathlib import Path
import numpy as np, pandas as pd
FEATURES = [f"V{i}" for i in range(1,29)]
COLUMNS = ["Time"]+FEATURES+["Amount","Class"]

def validate(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols: raise ValueError(f"Missing columns: {missing_cols}")
    return {"rows": int(len(df)), "missing_values": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "fraud_rate": float(df["Class"].mean()) if len(df) else 0.0,
            "class_counts": {int(k): int(v) for k,v in df["Class"].value_counts().sort_index().items()}}

def generate_synthetic_demo(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC demonstration data mimicking creditcardfraud schema. NOT real transactions."""
    rng = np.random.default_rng(seed)
    n_fraud = max(30, int(n*0.004))
    n_legit = n - n_fraud
    legit = rng.normal(0,1,size=(n_legit,28)); fraud = rng.normal(0,1,size=(n_fraud,28))
    # shift discriminative anonymized features for fraud (documented simulation choice)
    fraud[:, [9,11,13,16]] -= rng.uniform(2.0,4.0,size=(n_fraud,4))  # V10,V12,V14,V17
    fraud[:, [3,6]] += rng.uniform(1.0,2.5,size=(n_fraud,2))
    X = np.vstack([legit,fraud]); y = np.array([0]*n_legit+[1]*n_fraud)
    amt_legit = rng.lognormal(3.2,1.0,n_legit); amt_fraud = rng.lognormal(4.0,1.2,n_fraud)
    df = pd.DataFrame(X, columns=FEATURES)
    df["Time"] = rng.uniform(0,172800,n); df["Amount"] = np.concatenate([amt_legit,amt_fraud]); df["Class"]=y
    return df[COLUMNS].sample(frac=1, random_state=seed).reset_index(drop=True)

def load_dataset(raw_path: Path|None=None, demo_path: Path|None=None) -> tuple[pd.DataFrame,str]:
    if raw_path and Path(raw_path).exists():
        df = pd.read_csv(raw_path); validate(df); return df, "kaggle_creditcardfraud (observed)"
    if demo_path and Path(demo_path).exists():
        return pd.read_csv(demo_path), "SYNTHETIC_DEMO (simulated, not real transactions)"
    df = generate_synthetic_demo()
    if demo_path: Path(demo_path).parent.mkdir(parents=True, exist_ok=True); df.to_csv(demo_path,index=False)
    return df, "SYNTHETIC_DEMO (simulated, not real transactions)"
