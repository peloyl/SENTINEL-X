
import sys; sys.path.insert(0,".")
from src.data.loader import generate_synthetic_demo
from pathlib import Path
df=generate_synthetic_demo(20000)
Path("data/samples").mkdir(parents=True,exist_ok=True); df.to_csv("data/samples/demo_transactions.csv",index=False)
print(f"SYNTHETIC demo saved: {len(df)} rows, fraud={int(df.Class.sum())} ({df.Class.mean():.3%}) - NOT real transactions")
