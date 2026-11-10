"""Research evaluation helpers for chronological held-out experiments."""
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score, precision_recall_fscore_support

def forecast_metrics(y_true,y_pred):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float)
    if y.shape!=p.shape or not len(y): raise ValueError("y_true/y_pred shape mismatch or empty")
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)),"n":int(len(y))}

def anomaly_metrics(y_true,scores,threshold):
    y=np.asarray(y_true,dtype=int); s=np.asarray(scores,dtype=float)
    if y.shape!=s.shape or not len(y): raise ValueError("label/score shape mismatch or empty")
    pred=(s>=threshold).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    result={"precision":float(precision),"recall":float(recall),"f1":float(f1),
            "threshold":float(threshold),"n":int(len(y))}
    if len(np.unique(y))==2:
        result["roc_auc"]=float(roc_auc_score(y,s)); result["pr_auc"]=float(average_precision_score(y,s))
    return result

def policy_metrics(records):
    if not records: return {}
    n=len(records); calls=sum(bool(r.get("llm_invoked")) for r in records)
    lat=[float(r["decision_latency_ms"]) for r in records if "decision_latency_ms" in r]
    sla=[bool(r.get("sla_met")) for r in records if "sla_met" in r]
    return {"samples":n,"llm_calls":int(calls),"llm_call_rate":calls/n,
            "mean_decision_latency_ms":float(np.mean(lat)) if lat else None,
            "p95_decision_latency_ms":float(np.percentile(lat,95)) if lat else None,
            "sla_compliance":float(np.mean(sla)) if sla else None,
            "latency_measurement":"wall-clock measurements only" if lat else "not measured"}
