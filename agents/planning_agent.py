from typing import TypedDict,Optional,Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY

llm = ChatGroq(
    model = "llama-3.1-8b-instant",#llama-3.3-70b-versatile
    api_key=GROQ_API_KEY,
    temperature=0.3,
    request_timeout=60
)

prompt = ChatPromptTemplate.from_template(

    """You are a disaster risk analyst preparing a briefing for a NOC duty officer.

EVENT CLASSIFICATION
Disaster type: {disaster_type}
Predicted severity: {severity} (model confidence: {confidence})
Key contributing factors: {factor_text}
{live_note}

RELEVANT OFFICIAL GUIDANCE (retrieved from IMD/NDMA documents)
{context}

TASK
Write a concise operational briefing (4-6 sentences) for a duty officer. Include:
1. What was predicted and why (reference the contributing factors in plain language)
2. What the official guidance above recommends for this severity level
3. Any immediate watch points or caveats

Do not invent guidance not present in the retrieved context above. If the context doesn't cover this specific severity level, say so explicitly rather than guessing."""

)

chain = prompt | llm

class DisasterState(TypedDict):

    disaster_type : Optional[Literal["cyclone","earthquake"]]
    severity : Optional[str]
    severity_confidence : Optional[float]
    shap_factors : Optional[dict]
    earthquake_is_live : Optional[bool]
    retrieved_context : Optional[str]
    retrieved_sources : Optional[list]

    briefing: Optional[str]

def format_shap_factors(shap_factors: dict, top_n: int = 3) -> str:
    top = sorted(shap_factors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return ", ".join(
        f"{name} ({'increases' if val > 0 else 'decreases'} risk, contribution {val:.2f})"
        for name, val in top
    )

def generate_briefing(state:DisasterState) -> DisasterState:

    live_note = ""
    if state["disaster_type"] == "earthquake" and state.get("earthquake_is_live") is False:
        live_note = (
            "IMPORTANT: No active earthquake was detected near this location. "
            "This is a background regional seismic risk profile, not a live event. "
            "Phrase the briefing accordingly — do not imply an earthquake is currently happening."
        )

    response = chain.invoke(
        {
            "disaster_type":state["disaster_type"],
            "severity":state["severity"],
            "confidence": f"{state['severity_confidence']:.0%}",
            "factor_text": format_shap_factors(state["shap_factors"]),
            "context": state.get("retrieved_context") or "No specific guidance retrieved for this case.",
            "live_note": live_note
        }
    )

    state["briefing"] = response.content
    return state

if __name__ == "__main__":
    test_state: DisasterState = {
        "disaster_type": "cyclone",
        "severity": "Severe",
        "severity_confidence": 0.81,
        "shap_factors": {
            "initial_wind": 0.42,
            "genesis_lat": -0.18,
            "month": 0.05,
        },
        "earthquake_data_is_live": None,
        "retrieved_context": (
            "Severe Cyclonic Storms are characterized by wind speeds of 89-117 kmph. "
            "IMD issues Cyclone Warning (Red alert) 24 hours before expected landfall, "
            "requiring immediate evacuation of vulnerable coastal areas."
        ),
        "retrieved_sources": [{"source": "IMD", "file": "IMD Cyclone.pdf", "page": 12}],
        "briefing": None,
    }

    result = generate_briefing(test_state)
    print(result["briefing"])