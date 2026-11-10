"""Select escalation threshold on validation data only; never inspect test labels."""
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from config import THRESHOLD_PATH, THRESHOLD_SEARCH_LOW, THRESHOLD_SEARCH_HIGH, THRESHOLD_STEPS

def optimize_threshold(validation_scores, validation_labels, save=True):
    scores=np.asarray(validation_scores,dtype=float); labels=np.asarray(validation_labels,dtype=int)
    if scores.ndim!=1 or labels.shape!=scores.shape or not len(scores): raise ValueError("scores and labels must be equal non-empty vectors")
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)) or not np.isin(labels,[0,1]).all():
        raise ValueError("scores must be [0,1] and labels binary")
    candidates=np.linspace(THRESHOLD_SEARCH_LOW,THRESHOLD_SEARCH_HIGH,THRESHOLD_STEPS)
    rows=[]
    for threshold in candidates:
        pred=(scores>=threshold).astype(int)
        rows.append({"threshold":float(threshold),"f1":float(f1_score(labels,pred,zero_division=0)),
                     "precision":float(precision_score(labels,pred,zero_division=0)),
                     "recall":float(recall_score(labels,pred,zero_division=0))})
    best=max(rows,key=lambda r:(r["f1"],r["precision"],r["threshold"]))
    payload={"threshold":best["threshold"],"validation_metrics":best,"search":rows,
             "selection_split":"validation","test_set_used":False}
    if save:
        THRESHOLD_PATH.parent.mkdir(parents=True,exist_ok=True)
        THRESHOLD_PATH.write_text(json.dumps(payload,indent=2))
    return payload
