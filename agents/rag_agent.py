from typing import TypedDict,Optional,Literal
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTOR_DB_DIR = "rag/vector_db"
#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

class DisasterState(TypedDict):

    disaster_type : Optional[Literal["cyclone","earthquake"]]
    severity : Optional[str]
    severity_confidence : Optional[float]
    shap_factors : Optional[dict]
    earthquake_data_is_live: Optional[bool]

    retrieved_context : Optional[str]
    retrieved_sources : Optional[list]

def build_query(state:DisasterState) -> str:

    disaster_type = state["disaster_type"]
    severity = state["severity"]

    top_factors = sorted(
        state["shap_factors"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:3]

    factor_names = ", ".join(f"{name}" for name, _ in top_factors)

    return (
        f"Guidance and response protocol for {severity} severity {disaster_type}, "
        f"driven by {factor_names}"
    )

def retrieve_guidance(state:DisasterState,k:int = 6)->DisasterState:

    query = build_query()

    results = vector_db.similarity_search(
        query,
        k=k,
        filter={"disaster_type": state["disaster_type"].capitalize()}
    )

    if not results:
        state["retrieved_context"] = ""
        state["retrieved_sources"] = []

        return state
    
    context_chunks = []
    sources = []

    for doc in results:
        context_chunks.append(doc.page_content)
        sources.append(
            {
                "source":doc.metadata.get("source"),
                "file":doc.metadata.get("file"),
                "page":doc.metadata.get("page")
            }
        )

    state["retrieved_context"] = "\n\n--\n\n".join(context_chunks)
    state["retrieved_sources"] = sources

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
            "season": 0.02,
        },
        "earthquake_data_is_live": None,
        "retrieved_context": None,
        "retrieved_sources": None,
    }

    result = retrieve_guidance(test_state)
    print("Query used:", build_query(test_state))
    print("\n--- Retrieved context ---\n")
    print(result["retrieved_context"][:1000])
    print("\n--- Sources ---")
    for s in result["retrieved_sources"]:
        print(s)