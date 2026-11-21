
def generate_reasoning(risk: dict, shap_result: dict, rule_result: dict, velocity: dict, amount: float) -> str:
    top=[c for c in shap_result.get("contributions",[])[:3]]
    parts=[f"The transaction has been classified as {risk['level']} risk (final risk score {risk['final_score']:.2f}).",
           f"The model score was {risk['components']['model']:.2f}."]
    if top: parts.append("The most influential model features were "+", ".join(f"{c['feature']} ({c['shap_value']:+.2f})" for c in top)+". These are model contributions (SHAP), not probabilities or causal proof.")
    if amount>=1000: parts.append(f"The transaction amount (${amount:,.2f}) is high relative to the available reference distribution.")
    if rule_result.get("triggered"): parts.append(f"{len(rule_result['triggered'])} rule indicator(s) were triggered: "+", ".join(t["rule_id"] for t in rule_result["triggered"])+".")
    else: parts.append("No rule indicators were triggered.")
    if velocity.get("available"): parts.append(f"Velocity: {velocity['count_in_window']} transaction(s) in the configured window.")
    elif velocity.get("simulated"): parts.append("Velocity evidence is SIMULATED for demonstration and is not observed user behavior.")
    else: parts.append("Behavioral/velocity history is UNAVAILABLE for this dataset, so no behavioral claim is made.")
    parts.append(risk["recommendation"])
    return " ".join(parts)

def evidence_labels(shap_result, rule_result, velocity):
    return {"model_derived":[c["feature"] for c in shap_result.get("contributions",[])[:5]],
            "rule_derived":[t["rule_id"] for t in rule_result.get("triggered",[])],
            "observed_behavioral": velocity if velocity.get("available") else [],
            "simulated": velocity if velocity.get("simulated") else [],
            "unavailable": [] if velocity.get("available") else ["per-user 90-day history","device","IP/geolocation"]}
