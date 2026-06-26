import os
from typing import Optional,List
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse

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
    report_url : Optional[str] = None


REPORT_DIR = os.path.abspath("report")

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
    
    report_path = result.get("report_path")
    report_url = f"/report/{os.path.basename(report_path)}" if report_path else None
    
    return DisasterResponse(
        is_valid=result.get("is_valid", False),
        validation_errors=result.get("validation_errors", []),
        disaster_type=result.get("disaster_type"),
        severity=result.get("severity"),
        severity_confidence=result.get("severity_confidence"),
        briefing=result.get("briefing"),
        report_path=result.get("report_path"),
        report_url=report_url
    )

@app.get("reports/{filename}")
def get_report(filename:str):

    safe_path = os.path.normpath(os.path.join(REPORT_DIR,filename))

    if not safe_path.startswith(REPORT_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(safe_path,media_type="application/pdf",filename=filename)

@app.get("")