
import json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.features.engineering import Preprocessor, MODEL_FEATURES

def split_time_aware(df, test_size=0.2, valid_size=0.2, seed=42):
    d=df.sort_values("Time").reset_index(drop=True)
    n=len(d); n_test=int(n*test_size); n_valid=int(n*valid_size)
    test=d.iloc[n-n_test:]; rest=d.iloc[:n-n_test]; valid=rest.iloc[len(rest)-n_valid:]; train=rest.iloc[:len(rest)-n_valid]
    return train, valid, test

def evaluate(y, p, thr=0.5):
    pred=(p>=thr).astype(int)
    return {"pr_auc": float(average_precision_score(y,p)), "roc_auc": float(roc_auc_score(y,p)),
            "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)), "confusion_matrix": confusion_matrix(y,pred).tolist(), "threshold": thr}

def pick_threshold(y,p):
    best=(0.5,0)
    for thr in np.linspace(0.05,0.9,18):
        f=f1_score(y,(p>=thr).astype(int),zero_division=0)
        if f>best[1]: best=(float(thr),float(f))
    return best[0]

def train_all(df, out_dir: Path, seed=42, use_smote=False):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train,valid,test=split_time_aware(df,seed=seed)
    pre=Preprocessor().fit(train)
    Xtr,_=pre.transform(train); Xva,_=pre.transform(valid); Xte,_=pre.transform(test)
    ytr,yva,yte=train["Class"].values,valid["Class"].values,test["Class"].values
    if use_smote:
        from imblearn.over_sampling import SMOTE
        Xtr,ytr=SMOTE(random_state=seed).fit_resample(Xtr,ytr)  # inside training pipeline only
    spw=float((ytr==0).sum()/max(1,(ytr==1).sum()))
    lr=LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr,ytr)
    xgb=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,
                      scale_pos_weight=spw, eval_metric="aucpr", random_state=seed, n_jobs=2)
    xgb.fit(Xtr,ytr)
    thr=pick_threshold(yva, xgb.predict_proba(Xva)[:,1])
    report={"dataset_rows":int(len(df)),"train_rows":int(len(train)),"valid_rows":int(len(valid)),"test_rows":int(len(test)),
            "scale_pos_weight":spw,"chosen_threshold":thr,
            "logistic_regression_test":evaluate(yte,lr.predict_proba(Xte)[:,1],thr),
            "xgboost_valid":evaluate(yva,xgb.predict_proba(Xva)[:,1],thr),
            "xgboost_test":evaluate(yte,xgb.predict_proba(Xte)[:,1],thr),
            "features":MODEL_FEATURES,"model_version":"xgb-v1","smote_used":bool(use_smote)}
    joblib.dump({"model":xgb,"preprocessor":pre,"threshold":thr,"features":MODEL_FEATURES,"version":"xgb-v1"}, out_dir/"sentinelx_xgb.joblib")
    joblib.dump({"model":lr,"preprocessor":pre,"features":MODEL_FEATURES,"version":"lr-v1"}, out_dir/"sentinelx_lr.joblib")
    (out_dir/"evaluation_report.json").write_text(json.dumps(report,indent=2))
    return report
