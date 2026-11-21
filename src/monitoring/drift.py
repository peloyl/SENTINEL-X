
import numpy as np, pandas as pd
def psi(expected, actual, bins=10):
    """Population Stability Index. Data drift only - NOT concept drift."""
    qs=np.quantile(expected, np.linspace(0,1,bins+1)); qs=np.unique(qs)
    if len(qs)<3: return 0.0
    e=np.histogram(expected,bins=qs)[0]/max(1,len(expected)); a=np.histogram(actual,bins=qs)[0]/max(1,len(actual))
    e=np.clip(e,1e-4,None); a=np.clip(a,1e-4,None); return float(np.sum((a-e)*np.log(a/e)))
def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[dict]:
    out=[]
    for f in features:
        v=psi(reference[f].dropna().values, current[f].dropna().values)
        out.append({"feature":f,"psi":v,"status":"significant" if v>0.25 else "moderate" if v>0.1 else "stable"})
    return out
