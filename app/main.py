
"""SENTINEL-X - Financial Fraud Detection, Reasoning & Intelligence Platform
Run: streamlit run app/main.py
Pages are implemented in this single entrypoint (Streamlit multipage via sidebar radio)
to keep deployment simple and avoid path issues."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, streamlit as st, plotly.express as px
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="SENTINEL-X", layout="wide", page_icon="🛡️")
from app.components.style import apply_style; apply_style()
st.sidebar.title("🛡️ SENTINEL-X"); st.sidebar.caption("Detect the Pattern. Explain the Risk.")
page=st.sidebar.radio("Navigate", ["Overview","Transaction Investigation","Analyst Workflow","Reports & Analytics","Audit Logs"])

@st.cache_data
def load_data():
    from src.data.loader import load_dataset
    return load_dataset(ROOT/"data/raw/creditcard.csv", ROOT/"data/samples/demo_transactions.csv")

def get_predictor():
    p=ROOT/"models/sentinelx_xgb.joblib"
    if not p.exists(): return None
    from src.models.predict import Predictor
    return Predictor(p)

df, source = load_data()
st.sidebar.info(f"Data source: {source}")
pred=get_predictor()
if pred is None: st.sidebar.warning("No trained model. Run: python scripts/train_model.py")

# score a sample (cached) for overview - real model outputs, never hardcoded
@st.cache_data
def scored_sample(n=2000):
    if pred is None: return None
    from src.rules.engine import RuleEngine; from src.risk.engine import aggregate; from src.velocity.engine import VelocityEngine
    re_=RuleEngine(ROOT/"configs/rules.yaml"); ve=VelocityEngine()
    sample=df.head(n).copy(); scores=[]
    for _,row in sample.iterrows():
        ms=pred.score(row.to_dict()); r=re_.evaluate({**row.to_dict(),"hour":int((row.Time//3600)%24)})
        v=ve.score([], row.Time)  # no per-user history in this dataset -> unavailable, score 0
        a=aggregate(ms, r["score"], v["score"]); scores.append((ms,a["final_score"],a["level"]))
    sample["model_score"]=[s[0] for s in scores]; sample["final_score"]=[s[1] for s in scores]; sample["risk_level"]=[s[2] for s in scores]
    return sample

scored=scored_sample()

if page=="Overview":
    st.title("Overview")
    if scored is None: st.warning("Train the model first: `python scripts/train_model.py`")
    else:
        c=st.columns(5)
        c[0].metric("Total Transactions (sample)", f"{len(scored):,}")
        c[1].metric("Flagged (MEDIUM+HIGH)", f"{int((scored.risk_level!='LOW').sum()):,}")
        c[2].metric("High-Risk", f"{int((scored.risk_level=='HIGH').sum()):,}")
        c[3].metric("Fraud Rate (labels, sample)", f"{scored.Class.mean():.2%}")
        c[4].metric("Avg Risk Score", f"{scored.final_score.mean():.3f}")
        st.plotly_chart(px.histogram(scored,x="risk_level",color="risk_level",title="Risk Distribution"),use_container_width=True)
        st.plotly_chart(px.histogram(scored,x="Amount",nbins=50,title="Transaction Amount Distribution"),use_container_width=True)
        st.subheader("Recent Suspicious Transactions")
        st.dataframe(scored.sort_values("final_score",ascending=False).head(20)[["Time","Amount","Class","model_score","final_score","risk_level"]],use_container_width=True)

elif page=="Transaction Investigation":
    st.title("Transaction Investigation")
    if pred is None: st.warning("Train the model first.")
    else:
        idx=st.number_input("Transaction index (row in dataset)",0,len(df)-1,0)
        row=df.iloc[int(idx)].to_dict()
        from src.rules.engine import RuleEngine; from src.risk.engine import aggregate; from src.velocity.engine import VelocityEngine
        from src.explainability.shap_explainer import ShapExplainer; from src.reasoning.engine import generate_reasoning, evidence_labels
        from src.features.engineering import Preprocessor
        ms=pred.score(row); feats={**row,"hour":int((row["Time"]//3600)%24)}
        r=RuleEngine(ROOT/"configs/rules.yaml").evaluate(feats); v=VelocityEngine().score([],row["Time"]); risk=aggregate(ms,r["score"],v["score"])
        c=st.columns(4); c[0].metric("Amount",f"${row['Amount']:,.2f}"); c[1].metric("Model Score",f"{ms:.3f}"); c[2].metric("Final Risk",f"{risk['final_score']:.3f}"); c[3].metric("Risk Level",risk["level"])
        st.write(f"**Ground-truth label (dataset only):** {'Fraud' if row['Class']==1 else 'Legitimate'} | **Model version:** {pred.version}")
        # SHAP
        X,_=pred.pre.transform(pd.DataFrame([row])); expl=ShapExplainer(pred.model,pred.features); res=expl.explain(X[pred.features])
        contrib=pd.DataFrame(res["contributions"])
        st.plotly_chart(px.bar(contrib.head(12),x="shap_value",y="feature",orientation="h",title=f"Feature Contributions ({res['method']}) - waterfall data"),use_container_width=True)
        st.info("SHAP values are model contributions, not probabilities or causal explanations.")
        st.subheader("Human-Readable Reasoning"); st.write(generate_reasoning(risk,res,r,v,row["Amount"]))
        st.subheader("Triggered Rules"); st.write(r["triggered"] if r["triggered"] else "None triggered.")
        st.subheader("Evidence Classification"); st.json(evidence_labels(res,r,v))
        st.caption("Unavailable in this dataset: real IP, device, geolocation, 90-day per-user history.")

elif page=="Analyst Workflow":
    st.title("Analyst Workflow (simulated - not a bank payment control)")
    from src.audit.db import AuditDB
    db=AuditDB(ROOT/"data/sentinelx.db")
    tx=st.number_input("Transaction ID / index",0,len(df)-1,0)
    prev="UNKNOWN" if scored is None else str(scored.iloc[int(tx)]["risk_level"])
    st.write(f"Previous risk assessment: **{prev}**")
    action=st.selectbox("Action",["Mark as Legitimate","Confirm Fraud","Escalate for Review"])
    comment=st.text_input("Optional comment")
    if st.button("Save Decision"):
        db.save_decision(int(tx),action,prev,comment); st.success("Decision saved to SQLite with timestamp."); st.rerun()
    st.dataframe(db.decisions(),use_container_width=True)

elif page=="Reports & Analytics":
    st.title("Reports & Analytics")
    from src.audit.db import AuditDB; db=AuditDB(ROOT/"data/sentinelx.db"); counts=db.counts()
    c=st.columns(4); c[0].metric("Analyst Decisions",counts["decisions"]); c[1].metric("Confirmed Fraud (analyst)",counts["confirmed_fraud"]); c[2].metric("Marked Legitimate",counts["legitimate"]); c[3].metric("Escalated",counts["escalated"])
    st.caption("Analyst actions are NOT ground truth and are never used to compute model accuracy.")
    rep=ROOT/"models/evaluation_report.json"
    if rep.exists(): st.subheader("Model Evaluation (test split, real labels)"); st.json(json.loads(rep.read_text()))
    else: st.warning("No evaluation report yet - train the model.")
    if scored is not None:
        from sklearn.metrics import confusion_matrix
        st.subheader("False Positives on Scored Sample (vs dataset labels)")
        pred_label=(scored.final_score>=0.65).astype(int)
        st.write(confusion_matrix(scored.Class,pred_label).tolist()); st.caption("Rows=true [legit, fraud], Cols=predicted [legit, fraud] at HIGH threshold 0.65 on final risk.")

elif page=="Audit Logs":
    st.title("Audit Logs")
    from src.audit.db import AuditDB; db=AuditDB(ROOT/"data/sentinelx.db"); ev=db.audit_events()
    q=st.text_input("Filter (contains)")
    if q: ev=ev[ev.astype(str).apply(lambda r: q.lower() in " ".join(r).lower(),axis=1)]
    st.dataframe(ev,use_container_width=True)
    st.download_button("Export Audit CSV", ev.to_csv(index=False), "sentinelx_audit.csv", "text/csv")
    st.caption("Append-only at application level for MVP. Not tamper-proof or regulatory-certified.")
