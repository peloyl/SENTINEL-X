
import sys, json; sys.path.insert(0,".")
from pathlib import Path
from src.data.loader import load_dataset, validate
from src.models.train import train_all
df, source = load_dataset(Path("data/raw/creditcard.csv"), Path("data/samples/demo_transactions.csv"))
print("Source:", source); print(validate(df))
report=train_all(df, Path("models"))
print(json.dumps(report,indent=2))
