
import pandas as pd, pytest
from src.data.loader import generate_synthetic_demo, validate
from src.features.engineering import Preprocessor, MODEL_FEATURES
from src.rules.engine import RuleEngine
from src.risk.engine import aggregate
from src.velocity.engine import VelocityEngine
from src.reasoning.engine import generate_reasoning
from src.audit.db import AuditDB
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_validation_and_demo_labels():
    df=generate_synthetic_demo(2000); info=validate(df); assert info["rows"]==2000 and 0<info["fraud_rate"]<0.05

def test_preprocessing_no_leakage_shape():
    df=generate_synthetic_demo(1000); pre=Preprocessor().fit(df.iloc[:800]); X,_=pre.transform(df.iloc[800:])
    assert list(X.columns)==MODEL_FEATURES and len(X)==200

def test_rule_engine():
    e=RuleEngine(ROOT/"configs/rules.yaml"); r=e.evaluate({"Amount":6000,"V14":0,"V17":0,"V10":0,"hour":12})
    assert r["score"]>0 and any(t["rule_id"]=="R002" for t in r["triggered"])

def test_risk_aggregation():
    assert aggregate(1,1,1)["level"]=="HIGH" and aggregate(0,0,0)["level"]=="LOW"
    assert abs(aggregate(0.5,0.5,0.5)["final_score"]-0.5)<1e-9

def test_velocity_unavailable_is_honest():
    v=VelocityEngine().score([], 1000); assert v["available"] is False and v["score"]==0.0
    v2=VelocityEngine().simulated_score(5); assert v2["simulated"] is True

def test_reasoning_mentions_unavailable():
    risk=aggregate(0.9,1.0,0.0); txt=generate_reasoning(risk,{"contributions":[{"feature":"V14","shap_value":-1.2}]},{"triggered":[]},{"available":False},2000)
    assert "UNAVAILABLE" in txt and "HIGH" in txt

def test_audit_db(tmp_path):
    db=AuditDB(tmp_path/"t.db"); db.save_decision(7,"Confirm Fraud","HIGH","checked")
    assert db.counts()["confirmed_fraud"]==1 and len(db.audit_events())==1
    db.save_feedback(7,1); db.log_prediction(7,0.9,0.8,"HIGH","xgb-v1")

def test_model_train_and_shap(tmp_path):
    from src.models.train import train_all
    df=generate_synthetic_demo(6000); rep=train_all(df, tmp_path)
    assert rep["xgboost_test"]["pr_auc"]>0.5
    from src.models.predict import Predictor; from src.explainability.shap_explainer import ShapExplainer
    import pandas as pd
    p=Predictor(tmp_path/"sentinelx_xgb.joblib"); s=p.score(df.iloc[0].to_dict()); assert 0<=s<=1
    X,_=p.pre.transform(pd.DataFrame([df.iloc[0].to_dict()])); res=ShapExplainer(p.model,p.features).explain(X[p.features])
    assert len(res["contributions"])==len(p.features)
