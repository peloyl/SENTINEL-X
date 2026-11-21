
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
FEATURES=[f"V{i}" for i in range(1,29)]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out["hour"]=(out["Time"]//3600)%24
    out["log_amount"]=np.log1p(out["Amount"].clip(lower=0))
    out["amount_z"]=0.0  # filled by scaler-fit step on train only to avoid leakage
    return out

MODEL_FEATURES = FEATURES+["log_amount","hour"]

class Preprocessor:
    """Fit ONLY on training data. Amount scaled; V-features already PCA-scaled in source data."""
    def __init__(self): self.scaler=StandardScaler(); self.amount_mean_=0.0; self.amount_std_=1.0
    def fit(self, df):
        d=add_features(df); self.scaler.fit(d[["log_amount"]])
        self.amount_mean_=float(df["Amount"].mean()); self.amount_std_=float(df["Amount"].std() or 1.0); return self
    def transform(self, df):
        d=add_features(df); d["log_amount"]=self.scaler.transform(d[["log_amount"]]).ravel()
        d["amount_z"]=(d["Amount"]-self.amount_mean_)/self.amount_std_
        return d[MODEL_FEATURES], d
