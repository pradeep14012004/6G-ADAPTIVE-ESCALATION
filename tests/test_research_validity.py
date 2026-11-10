import numpy as np
import pandas as pd
from modules.preprocessing import generate_dataset, clean_data, engineer_features, build_sequences, train_test_split_ts
from modules.confidence import normalise_error, escalation_score
from modules.anomaly_detector import train_isolation_forest, batch_scores
from modules.threshold_optimizer import optimize_threshold
from modules.qos import compute_qos, sla_violations

def test_generator_reproducible_and_labels():
    a=generate_dataset(1000,seed=7); b=generate_dataset(1000,seed=7)
    pd.testing.assert_frame_equal(a,b)
    assert a.anomaly_label.mean() == 0.05

def test_rolling_features_are_lagged():
    d=generate_dataset(50,seed=1)
    f=engineer_features(d)
    assert f.loc[10,"traffic_roll_mean_6"] == d.loc[4:9,"traffic_mbps"].mean()

def test_sequences_and_chronological_split():
    X,y=build_sequences(np.arange(100),10)
    Xtr,Xte,ytr,yte=train_test_split_ts(X,y,.2)
    assert len(Xtr)+len(Xte)==len(X)
    assert ytr[-1] < yte[0]
    assert Xte[0,0,0] < yte[0]

def test_error_normalization_is_not_unit_clipped_for_mbps():
    assert np.isclose(normalise_error(150,100,200),.25)

def test_escalation_score_validation():
    assert np.isclose(escalation_score(.2,.3,.5),.29)
    import pytest
    with pytest.raises(ValueError): escalation_score(1.2,0,0)

def test_iforest_scores_calibrated_on_train_only():
    rng=np.random.default_rng(2); X=rng.normal(size=(200,4))
    model=train_isolation_forest(X)
    scores=batch_scores(model,np.vstack([X[:3],np.ones((1,4))*10]))
    assert scores.shape==(4,)
    assert np.all((scores>=0)&(scores<=1))
    assert scores[-1] > scores[:3].mean()

def test_threshold_uses_validation_and_returns_reproducible_threshold():
    scores=np.array([.1,.2,.8,.9]); labels=np.array([0,0,1,1])
    result=optimize_threshold(scores,labels,save=False)
    assert result["selection_split"]=="validation"
    assert result["test_set_used"] is False
    assert result["validation_metrics"]["f1"]==1.0

def test_qos_score_bounded_and_sla_explicit():
    assert 0 <= compute_qos(80,99.99,5,"ICU Monitoring") <= 1
    assert sla_violations(20,99,10,"ICU Monitoring")["latency_violation"]
