"""Leakage-aware synthetic data generation and chronological preprocessing."""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH, SCALER_PATH, N_SAMPLES, RANDOM_SEED, SEQ_LEN, TEST_SPLIT

FEATURE_COLS = ["traffic_mbps","latency_ms","bandwidth_mbps","packet_loss","reliability","jitter_ms",
                "hour","day_of_week","is_peak","traffic_roll_mean_6","traffic_roll_std_6",
                "traffic_roll_mean_12","traffic_roll_std_12","traffic_roll_mean_24","traffic_roll_std_24",
                "traffic_diff1","traffic_diff2"]

def generate_dataset(n=N_SAMPLES, seed=RANDOM_SEED, anomaly_rate=0.05):
    if n <= 0 or not 0 <= anomaly_rate < 1: raise ValueError("invalid n or anomaly_rate")
    rng=np.random.default_rng(seed); t=np.arange(n)
    base=60+24*np.sin(2*np.pi*t/144)+8*np.sin(2*np.pi*t/(144*7))+0.001*t
    innovations=rng.normal(0,2.5,n)
    traffic=np.empty(n); traffic[0]=base[0]+rng.normal(0,3)
    for i in range(1,n): traffic[i]=0.72*traffic[i-1]+0.28*base[i]+innovations[i]
    labels=np.zeros(n,dtype=int); count=int(round(n*anomaly_rate))
    if count:
        idx=rng.choice(n,count,replace=False); labels[idx]=1
        traffic[idx]+=rng.choice([-1,1],count)*rng.uniform(35,75,count)
    traffic=np.clip(traffic,5,220)
    latency=np.clip(1+0.055*traffic+rng.normal(0,.35,n),.2,30)
    bandwidth=np.clip(.88*traffic+rng.normal(0,2,n),1,250)
    packet_loss=np.clip(.0008*traffic+rng.normal(0,.012,n),0,.25)
    reliability=np.clip(100-packet_loss*10,95,100)
    jitter=np.clip(.12*latency+rng.normal(0,.08,n),0,10)
    devices=rng.choice(["Remote Surgery","ICU Monitoring","Wearable Devices","Emergency Ambulance"],n,p=[.15,.35,.35,.15])
    return pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=n,freq="10min"),
        "traffic_mbps":traffic,"latency_ms":latency,"bandwidth_mbps":bandwidth,
        "packet_loss":packet_loss,"reliability":reliability,"jitter_ms":jitter,
        "device_type":devices,"anomaly_label":labels})

def clean_data(df):
    out=df.copy().drop_duplicates().reset_index(drop=True)
    out["timestamp"]=pd.to_datetime(out["timestamp"])
    numeric=out.select_dtypes(include=[np.number]).columns
    out[numeric]=out[numeric].replace([np.inf,-np.inf],np.nan)
    out[numeric]=out[numeric].ffill().bfill()
    if out[numeric].isna().any().any(): raise ValueError("unresolved numeric missing values")
    return out

def engineer_features(df):
    out=df.copy(); ts=pd.to_datetime(out["timestamp"])
    out["hour"]=ts.dt.hour; out["day_of_week"]=ts.dt.dayofweek
    out["is_peak"]=((out["hour"]>=8)&(out["hour"]<=20)).astype(int)
    # Shift first: every row's rolling features use past observations only.
    past=out["traffic_mbps"].shift(1)
    for w in (6,12,24):
        out[f"traffic_roll_mean_{w}"]=past.rolling(w,min_periods=1).mean().fillna(out["traffic_mbps"].iloc[0])
        out[f"traffic_roll_std_{w}"]=past.rolling(w,min_periods=2).std().fillna(0)
    out["traffic_diff1"]=out["traffic_mbps"].diff().fillna(0)
    out["traffic_diff2"]=out["traffic_diff1"].diff().fillna(0)
    return out

def scale_features(df,fit=True,scaler=None):
    out=df.copy()
    if fit:
        scaler=MinMaxScaler().fit(out[FEATURE_COLS])
        SCALER_PATH.parent.mkdir(parents=True,exist_ok=True); joblib.dump(scaler,SCALER_PATH)
    else:
        scaler=scaler or joblib.load(SCALER_PATH)
    out.loc[:,FEATURE_COLS]=scaler.transform(out[FEATURE_COLS])
    return out,scaler

def build_sequences(series,seq_len=SEQ_LEN):
    series=np.asarray(series,dtype=float)
    if seq_len<1 or len(series)<=seq_len: raise ValueError("series must be longer than positive seq_len")
    X=np.array([series[i:i+seq_len] for i in range(len(series)-seq_len)])
    y=np.array([series[i+seq_len] for i in range(len(series)-seq_len)])
    return X[...,None],y

def train_test_split_ts(X,y,test_ratio=TEST_SPLIT):
    split=int(len(X)*(1-test_ratio))
    if split<1 or split>=len(X): raise ValueError("test_ratio must leave non-empty train and test sets")
    return X[:split],X[split:],y[:split],y[split:]

def run_pipeline(save=True):
    df=engineer_features(clean_data(generate_dataset()))
    split=int(len(df)*(1-TEST_SPLIT))
    if split<=SEQ_LEN: raise ValueError("not enough training observations")
    # Fit scaler on training rows only; transform test using frozen training scaler.
    scaler=MinMaxScaler().fit(df.iloc[:split][FEATURE_COLS])
    scaled=df.copy(); scaled.loc[:,FEATURE_COLS]=scaler.transform(df[FEATURE_COLS])
    SCALER_PATH.parent.mkdir(parents=True,exist_ok=True); joblib.dump(scaler,SCALER_PATH)
    if save:
        DATA_PATH.parent.mkdir(parents=True,exist_ok=True); scaled.to_csv(DATA_PATH,index=False)
    X,y=build_sequences(scaled["traffic_mbps"].to_numpy())
    Xtr,Xte,ytr,yte=train_test_split_ts(X,y)
    return scaled,scaler,Xtr,Xte,ytr,yte
