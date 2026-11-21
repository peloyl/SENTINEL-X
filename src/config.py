
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
def load_settings() -> dict:
    return yaml.safe_load((ROOT/"configs/settings.yaml").read_text())
def load_rules_config() -> dict:
    return yaml.safe_load((ROOT/"configs/rules.yaml").read_text())
