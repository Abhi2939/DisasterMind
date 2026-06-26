from typing import Optional,List
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException

from orchestrator import disaster_graph


app = FastAPI()

class DisasterRequest(BaseModel):

    location : str
    initial_wind: Optional[float] = None
    pressure_hpa : Optional[float] = None
    depth : Optional[float] = None
    subbasin : Optional[str] = None 

class DisasterResponse(BaseModel):

    is_valid : bool
    validation_errors: List[str]
    disaster_type: Optional[str] = None
    severity: Optional[float] = None
    severity_confidence: Optional[float] = None
    breifing : Optional[str] = None
    report_path : Optional[str] = None

@app.get("/")
def root():
    return {"status":"DisasterMind API is Running"}

@app.post("/predict",response_model = DisasterResponse)
def predict(request : DisasterResponse):

    raw_input = request.model_dump(exclude_none=True)

    try:
        result =  disaster_graph.invoke({"raw_input":raw_input})
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Pipeline error{e}")
    
    return DisasterResponse(
        is_valid=result.get("is_valid", False),
        validation_errors=result.get("validation_errors", []),
        disaster_type=result.get("disaster_type"),
        severity=result.get("severity"),
        severity_confidence=result.get("severity_confidence"),
        briefing=result.get("briefing"),
        report_path=result.get("report_path"),
    )


