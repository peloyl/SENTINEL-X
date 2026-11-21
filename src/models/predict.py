
import joblib
from pathlib import Path
class Predictor:
    def __init__(self, bundle_path):
        if not Path(bundle_path).exists(): raise FileNotFoundError("Trained model not found. Run: python scripts/train_model.py")
        b=joblib.load(bundle_path); self.model=b["model"]; self.pre=b["preprocessor"]; self.threshold=b.get("threshold",0.5); self.features=b["features"]; self.version=b.get("version","unknown")
    def score(self, row) -> float:
        import pandas as pd
        X,_=self.pre.transform(pd.DataFrame([row])); return float(self.model.predict_proba(X[self.features])[:,1][0])
