
def aggregate(model_score: float, rule_score: float, velocity_score: float, weights=(0.6,0.2,0.2)) -> dict:
    wm,wr,wv=weights; total=wm+wr+wv; wm,wr,wv=wm/total,wr/total,wv/total
    final=wm*model_score+wr*rule_score+wv*velocity_score
    level="HIGH" if final>=0.65 else "MEDIUM" if final>=0.35 else "LOW"
    rec={"HIGH":"Escalate for investigation; do not approve automatically.","MEDIUM":"Analyst review recommended.","LOW":"No immediate action; routine monitoring."}[level]
    return {"final_score":float(final),"level":level,"recommendation":rec,
            "components":{"model":float(model_score),"rule":float(rule_score),"velocity":float(velocity_score)},
            "weights":{"model":wm,"rule":wr,"velocity":wv}}
