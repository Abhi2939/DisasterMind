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
def build_dataset():

    rows = []
    for case in TEST_SET:
        results = vector_db.similarity_search(
            case["question"],k=4,filter={"disaster_type":case["disaster_type"]}
        )

        retrieved_chunks = [doc.page_content for doc in results]

        rows.append(
            {
                "user_input":case["question"],
                "retrieved_context":retrieved_chunks,
                "response": " ".join(retrieved_chunks)[:500], 
                "reference": case["ground_truth"]
            }
        )

    return EvaluationDataset.from_list(rows)

def main():
    dataset = build_dataset()

    llm = LangchainLLMWrapper(
        ChatGroq(
            model = "llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0
        )
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(llm=llm),
            Faithfulness(llm=llm),
            FactualCorrectness(llm=llm),
            LLMContextPrecisionWithReference(llm=llm)
        ]
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)
    results.to_pandas().to_csv("rag_evaluation_results.csv", index=False)
    print("Saved to rag_evaluation_results.csv")


if __name__ == "__main__":
    main()