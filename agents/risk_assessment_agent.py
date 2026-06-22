import numpy as np
import pandas as pd
import joblib
import shap
from typing import Optional,TypedDict,Literal

#Loading Models

model1 = joblib.load("models/model1_cyclone_vs_earthquake.pkl")

model2 = joblib.load("models/model2_cyclone_severity.pkl")
model2_encoder = joblib.load("models/model2_encoder.pkl")
model2_le = joblib.load("models/model2_label_enocoder.pkl")

model3 = joblib.load("models/model3_earthquake_severity.pkl")
model3_encoder = joblib.load("models/model3_encoder.pkl")

model2_shap = shap.TreeExplainer(model2)
model3_shap = shap.TreeExplainer(model3.named_steps["clf"])

class DisasterState(TypedDict):

    latitude: float
    longitude: float
    month: int
    hour: Optional[int]
    year: Optional[int]
    subbasin: Optional[str]
    initial_wind: Optional[int]
    depth: Optional[float]
    earthquake_data_is_live: Optional[bool]

    disaster_type: Optional[Literal["earthquake","cyclone"]]
    severity: Optional[str]
    severity_confidence: Optional[int]
    shap_factors: Optional[dict]

# Routing for model 1
def route_disaster_type(state:DisasterState) -> DisasterState:

    month_sin = np.sin(2*np.pi*state["month"]/12)
    month_cos= np.cos(2*np.pi*state["month"]/12)

    X = pd.DataFrame([{
        "latitude":state["lat"],
        "longitutude":state["lon"],
        "month_sin":month_sin,
        "month_cos":month_cos
    }])

    pred = model1.predict(X)[0]
    state["disaster_type"] = "cyclone" if pred == 1 else "earthquake"
    return state

def pick_branch(state:DisasterState) -> str:
    return state["disaster_type"]

# Model2 :- Cyclone Severity
def get_month(month:int) -> str:
    if pd.isna(month) : return "Unknown"

    if month in [6,7,8,9] : return "Monsoon"
    elif month in [3,4,5] : return "PreMonsoon"
    elif month in [10,11] : return "PostMonsoon"
    else : return "Winter"

def cyclone_severity(state:DisasterState) -> DisasterState:

    raw = pd.DataFrame([{
        "season" : get_month(state["month"]),
        "subbasin" : state.get("subbasin") or "BB",
        "month":state["month"],
        "genesis_lat":state["latitude"],
        "genesis_lon":state["longitude"],
        "initial_wind_speed":state.get("initial_wind", np.nan)
    }])

    raw[["season","subbasin"]] = model2_encoder.transfrom(raw[["season", "subbasin"]].astype(str))

    pred = model2.predict(raw)[0]
    proba = model2.predict_proba(raw)[0]

    state["severity"] = model2_le.inverse_transform([pred])[0]
    state["severity_confidence"] = float(proba[pred])



