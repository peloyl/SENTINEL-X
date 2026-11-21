
import numpy as np
class ShapExplainer:
    """SHAP TreeExplainer for XGBoost; falls back to model feature_importances_ ranking if SHAP unavailable.
    SHAP values are model contributions, NOT probabilities and NOT causal explanations."""
    def __init__(self, model, feature_names):
        self.model=model; self.features=feature_names; self._explainer=None
        try:
            import shap; self._explainer=shap.TreeExplainer(model)
        except Exception: self._explainer=None
    def explain(self, X_row):
        if self._explainer is not None:
            vals=self._explainer.shap_values(X_row)
            vals=np.asarray(vals).reshape(-1)[:len(self.features)]
            base=float(np.asarray(self._explainer.expected_value).reshape(-1)[0]) if hasattr(self._explainer,"expected_value") else 0.0
        else:
            imp=getattr(self.model,"feature_importances_",np.zeros(len(self.features)))
            vals=(np.asarray(X_row).reshape(-1)*imp); base=0.0
        ranked=sorted([{"feature":f,"shap_value":float(v)} for f,v in zip(self.features,vals)], key=lambda d: abs(d["shap_value"]), reverse=True)
        return {"contributions":ranked,"base_value":base,"method":"shap" if self._explainer else "importance_fallback"}
    def global_importance(self):
        imp=getattr(self.model,"feature_importances_",None)
        if imp is None: return []
        return sorted([{"feature":f,"importance":float(v)} for f,v in zip(self.features,imp)], key=lambda d:d["importance"], reverse=True)
