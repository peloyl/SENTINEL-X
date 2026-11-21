
import yaml
from pathlib import Path
class RuleEngine:
    def __init__(self, rules_path):
        cfg=yaml.safe_load(Path(rules_path).read_text()); self.version=cfg.get("version","rules-v1"); self.rules=[r for r in cfg["rules"] if r.get("enabled",True)]
    def evaluate(self, features: dict):
        triggered=[]; total=0.0
        for r in self.rules:
            if r["feature"] not in features or features[r["feature"]] is None: continue
            v=float(features[r["feature"]]); cond=r["condition"]; thr=r["threshold"]
            hit = (v>thr) if cond==">" else (v<thr) if cond=="<" else (thr[0]<=v<=thr[1]) if cond=="between" else False
            if hit:
                triggered.append({"rule_id":r["id"],"feature":r["feature"],"value":v,"severity":float(r["severity"]),
                                  "explanation":r["template"].format(value=round(v,3)),"source":"rule"})
                total+=float(r["severity"])
        score=min(1.0, total/2.0) if triggered else 0.0  # normalized 0-1, capped to avoid double-count inflation
        return {"score":score,"triggered":triggered,"version":self.version}
