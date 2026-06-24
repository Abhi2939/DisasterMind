from typing import TypedDict,Optional,Literal
from groq import Groq
from config import GROQ_API_KEY

# client = Groq(api_key=GROQ_API_KEY)

# MODEL = "llama-3.3-70b-versatile"

class DisasterState(TypedDict):

    disaster_type : Optional[Literal["cyclone","earthquake"]]
    severity : Optional[str]
    severity_confidence : Optional[float]
    shap_factors : Optional[dict]
    earthquake_is_live : Optional[bool]
    retrieved_context : Optional[str]
    retrieved_sources : Optional[list]

    briefing: Optional[str]

