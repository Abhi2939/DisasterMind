import os
from datetime import datetime,timezone
from typing import TypedDict,Literal,Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

REPORT_DIR = "report"

class DisasterSate(TypedDict):

    latitude : float
    longitude : float
    month : int
    disaster_type : Optional[Literal["earthquake","cyclone"]]
    severity : Optional[str]
    severity_confidence : Optional[float]
    shap_factors : Optional[dict]
    earthquake_data_is_live : Optional[bool]
    retrieved_context : Optional[str]
    retrieved_sources : Optional[list]
    briefing : Optional[str]

    report_path : Optional[str]

def severity_color(severity:str) -> colors.Color :
    """Maps severity to a visual flag, loosely aligned with IMD colour codes."""
    high_risk = {"Severe", "Significant"}
    return colors.HexColor("#C0392B") if severity in high_risk else colors.HexColor("#1E8449")

