from agents.rag_agent import vector_db
from ragas import evaluate,EvaluationDataset
from ragas.metrics import LLMContextRecall,Faithfulness,FactualCorrectness,LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Test Question Answering 
TEST_SET = [
    {
        "question": "What wind speed range defines a Severe Cyclonic Storm?",
        "ground_truth": "FILL IN from IMD Cyclone SOP",
        "disaster_type": "Cyclone",
    },
    {
        "question": "What are the four stages of IMD's cyclone warning system?",
        "ground_truth": "FILL IN from IMD Cyclone SOP",
        "disaster_type": "Cyclone",
    },
    {
        "question": "What does a Red colour-coded cyclone warning indicate?",
        "ground_truth": "FILL IN from IMD Cyclone SOP",
        "disaster_type": "Cyclone",
    },
    {
        "question": "What is NDMA's recommended preparedness action for earthquake-prone zones?",
        "ground_truth": "FILL IN from NDMA Earthquake guidelines",
        "disaster_type": "Earthquake",
    },
    {
        "question": "How does NDMA classify earthquake risk zones in India?",
        "ground_truth": "FILL IN from NDMA Earthquake guidelines",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What immediate response actions does NDMA recommend during an earthquake?",
        "ground_truth": "FILL IN from NDMA Earthquake guidelines",
        "disaster_type": "Earthquake",
    },
]

