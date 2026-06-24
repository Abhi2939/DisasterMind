import os
from datetime import datetime,timezone
from typing import TypedDict,Literal,Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

REPORT_DIR = "report"

class DisasterState(TypedDict):

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


def generate_report(state:DisasterState) -> DisasterState:

    os.makedirs(REPORT_DIR,exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{REPORT_DIR}/incident_report_{state['disaster_type']}_{timestamp}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18)
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    caveat_style = ParagraphStyle(
        "Caveat", parent=normal_style, textColor=colors.HexColor("#B9770E"), fontSize=9
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize = letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch
    )

    story = []

    # header
    story.append(Paragraph("DisasterMind — Incident Risk Report",title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Generated: {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}",
        normal_style
    ))
    story.append(Spacer(1, 16))

    # classification summary table
    is_live = state.get("earthquake_data_is_live")
    live_label = ""

    if state["disaster_type"] == "earthquake":
        live_label = "Live event detected" if is_live else "Regional historical profile (no active event)"

    summary_data = [
        ["Disaster Type", state["disaster_type"].capitalize()],
        ["Location", f"{state['latitude']:.3f}, {state['longitude']:.3f}"],
        ["Predicted Severity", state["severity"]],
        ["Model Confidence", f"{state['severity_confidence']:.0%}"],
    ]

    if live_label:
        summary_data.append(["Data Status",live_label])

    table = Table(summary_data, colWidths=[2 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 2), (1, 2), severity_color(state["severity"])),
        ("TEXTCOLOR", (0, 2), (1, 2), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    story.append(Spacer(1, 16))

    if state["disaster_type"] == "earthquake" and is_live is False:
        story.append(
            Paragraph(
                "⚠ This assessment is based on a regional historical seismic profile, "
            "not a live detected event. Treat as background risk context only.",
            caveat_style
            )
        )
        story.append(Spacer(1, 12))

    # factors 
    story.append(Paragraph("Key Contributing Factor (SHAP)",heading_style))
    top_factors = sorted(state["shap_factors"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    factor_rows = [["Factor","Contribution","Direction"]]
    for name, val in top_factors:
        factor_rows.append([name, f"{val:.3f}", "↑ increases risk" if val > 0 else "↓ decreases risk"])
    factor_table = Table(factor_rows, colWidths=[2 * inch, 1.5 * inch, 2 * inch])
    factor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(factor_table)
    story.append(Spacer(1, 16))
    
    # briefing (from plannig agent)
    story.append(Paragraph("Operational Briefing", heading_style))
    story.append(Paragraph(state.get("briefing") or "No briefing generated.", normal_style))
    story.append(Spacer(1, 16))

    # sources (from RAG Agent)
    story.append(Paragraph("Guidance Sources", heading_style))
    sources = state.get("retrieved_sources") or []

    if sources:
        for s in sources:
            story.append(
                Paragraph(
                    f"• {s.get('source', 'Unknown')} — {s.get('file', '')}, page {s.get('page', '?')}",
                    normal_style
                )
            )
    else:
        story.append(
            Paragraph(
                "No specific sources documents retrieved.",normal_style
            )
        )

    doc.build(story)

    state["report_path"] = filename
    return state


if __name__ == "__main__":
    test_state: DisasterState = {
        "latitude": 18.5, "longitude": 86.0, "month": 5,
        "disaster_type": "cyclone",
        "severity": "Severe",
        "severity_confidence": 0.81,
        "shap_factors": {
            "initial_wind": 0.42, "genesis_lat": -0.18, "month": 0.05,
            "subbasin": 0.03, "season": -0.01,
        },
        "earthquake_data_is_live": None,
        "retrieved_context": "...",
        "retrieved_sources": [
            {"source": "IMD", "file": "IMD Cyclone.pdf", "page": 12},
            {"source": "NDMA", "file": "NDMA Cyclone.pdf", "page": 4},
        ],
        "briefing": (
            "A Severe Cyclonic Storm is predicted with 81% model confidence, primarily "
            "driven by elevated genesis-stage wind speed. Per IMD guidance, this severity "
            "level warrants a Cyclone Warning (Red alert) with evacuation of vulnerable "
            "coastal areas within 24 hours of expected landfall."
        ),
        "report_path": None,
    }

    result = generate_report(test_state)
    print("Report saved to:", result["report_path"])