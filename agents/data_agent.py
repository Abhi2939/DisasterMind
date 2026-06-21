from dotenv import load_dotenv
import os
import requests
from datetime import datetime,timezone,timedelta
from typing import TypedDict,Optional


load_dotenv()

OWM_API_KEY = os.environ["OWN_API_KEY"]


#geocoding
def geocode_location(location:str) -> dict:
    url = "https://api.openweathermap.org/geo/1.0/direct"
    resp = requests.get(url,params={"q":location,"limit":1,"appid":OWM_API_KEY},timeout=10)
    resp.raise_for_status()
    results = resp.json()
    
    if not results:
        raise ValueError(f"Could not geocode location: {location}")
    return {"latitude": results[0]["lat"], "longitude": results[0]["lon"]}

#OWM API
def fetch_current_waather(lat:float,lon:float) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url,params={lat:"lat",lon:"lon","appid":OWM_API_KEY,"units":"metric"},timeout=10)
    resp.raise_for_status()
    data = resp.json()
    wind_ms = data.get("wind",{}).get("speed")
    return {
        "wind_speed_kt": wind_ms * 1.94384 if wind_ms is not None else None,
        "pressure_hpa": data.get("main", {}).get("pressure"),
        "observed_at": datetime.fromtimestamp(data.get("dt", 0), tz=timezone.utc).isoformat(),
        "is_live": True,
    }

#USGS API
REGION_PROFILE = {
    "Himalayan": {"avg_depth": 35, "typical_mag": (4.5, 6.5)},
    "Western":   {"avg_depth": 20, "typical_mag": (4.0, 5.5)},
    "Southern":  {"avg_depth": 10, "typical_mag": (3.5, 5.0)},
    "Northeast": {"avg_depth": 45, "typical_mag": (4.5, 6.0)},
    "Central":   {"avg_depth": 15, "typical_mag": (3.5, 5.5)},
}

def get_seismic_region(lat:float,lon:float) -> str:
    if lat >= 28: return "Himalyan"
    elif lon <= 72 : return "Western"
    elif lat <= 15 : return 'Southern'
    elif lon >= 90 : return "Northeast"
    else : return "Central"


def fetch_live_earthquake(lat:float,lon:float,radius_km: int = 300,days : int = 30) -> Optional[dict]:

    url = "https://earthquake.usgs.gov/fdsnws/event/1.0/query"
    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "starttime": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
        "orderby": "time",
        "limit": 1,
    }

    resp = requests.get(url,params=params,timeout=10)
    resp.raise_for_status()

    features = resp.json().get("features",[])
    if not features:
        return None
    f = features[0]
    lon_e, lat_e, depth = f["geometry"]["coordinates"]
    return {
        "depth": depth,
        "magnitude": f["properties"]["mag"],
        "observed_at": datetime.fromtimestamp(f["properties"]["time"] / 1000, tz=timezone.utc).isoformat(),
        "is_live": True,
    }

def seismic_data(lat:float,lon:float) -> dict:
    live = fetch_live_earthquake(lat,lon)
    if live:
        return live
    
    region = get_seismic_region(lat,lon)
    profile = REGION_PROFILE[region]

    return {
        "depth": profile["avg_depth"],
        "magnitude": None,
        "region": region,
        "is_live": False,
    }

#Node 

class DisasterState(TypedDict):

    raw_input : dict
    longitude : Optional[float]
    latitude : Optional[float]
    month : Optional[int]
    hour : Optional[int]
    year : Optional[int]
    subbasin : Optional[str]
    initial_wind: Optional[float]
    pressure_hpa: Optional[float]
    depth: Optional[float]
    earthquake_data_is_live: Optional[bool]
    is_valid: bool
    validation_errors: list

def get_subbasin(lat:float,lon:float) -> Optional[str]:

    if 5<=lat<=22 and 80 <=lon<=100: return "BB"
    elif 5<=lat<=25 and 60<=lon<=78: return "AB"

    return None

def data_agent(state:DisasterState) -> DisasterState:

    location = state["raw_input"].get("location")
    if not location:
        state.update({"is_valid":False,"validation_errors":["Location not provided"]})
        return state
    
    try:
        geo = geocode_location(location)
    except Exception as e:
        state.update({"is_valid":False,"validation_errors":[str(e)]})
        return state
    lat,lon = geo["latitude"],geo["longitude"]
    now = datetime.now(timezone.utc)
    weather = fetch_current_waather(lat,lon)
    seismic = get_seismic_region(lat,lon)

    state.upadte({
        "latitude":lat,"longitude":lon,
        "month": now.month, "hour": now.hour, "year": now.year,
        "subbasin":get_subbasin(lat,lon),
        "initial_wind": weather["wind_speed_kt"],
        "pressure_hpa": weather["pressure_hpa"],
        "depth":seismic["depth"],
        "earthquake_data_is_live":seismic["is_live"],
        "is_valid":True,
        "validation_errors":[]
    })

    return state

