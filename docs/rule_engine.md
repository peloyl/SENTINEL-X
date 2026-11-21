
# Rule Engine
Rules live in `configs/rules.yaml`: id, feature, condition (>, <, between), threshold, severity (0-1), template, enabled.
Rule score = min(1, sum(triggered severities)/2) - capped to limit double-counting.
Add a rule by editing YAML only; no code change needed.
