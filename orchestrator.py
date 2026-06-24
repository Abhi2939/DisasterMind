from typing import TypedDict,Optional,Literal
from langgraph.graph import StateGraph,END

from agents.data_agent import data_agent
from agents.risk_assessment_agent import route_disaster_type,cyclone_severity,earthquake_severity
from agents.rag_agent import retrieve_guidance
from agents.planning_agent import generate_briefing
from agents.report_agent import generate_report

class DisasterState(TypedDict):

    raw_input : dict

    #data agent
    latitude: Optional[float]
    longitude: Optional[float]
    month: Optional[int]
    hour: Optional[int]
    year: Optional[int]
    subbasin: Optional[str]
    initial_wind: Optional[float]
    pressure_hpa: Optional[float]
    depth: Optional[float]
    earthquake_data_is_live: Optional[bool]
    is_valid: bool
    validation_errors: list

    #Risk assessment agent
    disaster_type: Optional[Literal["earthquake","cyclone"]]
    severity: Optional[str]
    severity_confidence: Optional[int]
    shap_factors: Optional[dict]

    #RAG agent 
    retrieved_context : Optional[str]
    retrieved_sources : Optional[list]

    #Planning agent
    briefing: Optional[str]

    #Report Agent
    report_path : Optional[str]

def validation_gate(state:DisasterState) -> str:
    return "route" if state["is_valid"] else "reject"

def pick_branch(state:DisasterState) -> str:
    return state["disaster_type"]

def reject_invalid(state: DisasterState) -> DisasterState:
    state["briefing"] = f"Input rejected: {', '.join(state['validation_errors'])}"
    return state

#Build Graph
graph = StateGraph(DisasterState)

graph.add_node("data_agent",data_agent)
graph.add_node("route",route_disaster_type)
graph.add_node("cyclone severity",cyclone_severity)
graph.add_node("earthquake severity",earthquake_severity)
graph.add_node("rag",retrieve_guidance)
graph.add_node("plan",generate_briefing)
graph.add_node("report",generate_report)
graph.add_node("reject",reject_invalid)

