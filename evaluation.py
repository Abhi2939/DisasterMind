from agents.rag_agent import vector_db
from ragas import evaluate,EvaluationDataset
from ragas.metrics import LLMContextRecall,Faithfulness,FactualCorrectness,LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Test Question Answering 
TEST_SET = [
    {
        "question": "What is NDMA's recommended preparedness action for earthquake-prone zones?",
        "ground_truth": "Repair deep plaster cracks in ceilings and foundations, anchor overhead lighting fixtures, fasten shelves securely to walls, secure water heaters and LPG cylinders by strapping them to walls or bolting to the floor, identify safe places indoors and outdoors, and have a disaster emergency kit ready with torch, first aid kit, food, water, and essential medicines.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "How does NDMA classify earthquake risk zones in India?",
        "ground_truth": "India is divided into four seismic zones: Zone II (low damage risk, MSK VI or less, 41.40% of geographical area), Zone III (moderate damage risk, MSK VII, 30.40%), Zone IV (high damage risk, MSK VIII, 17.30%), and Zone V (very high damage risk, MSK IX or more, 10.90%). Zones III, IV and V together cover 58.6% of India's land area and are classified as high risk areas.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What immediate response actions does NDMA recommend during an earthquake?",
        "ground_truth": "If indoors: DROP to the ground, take COVER under a sturdy table or furniture, and HOLD ON until shaking stops. Stay away from glass, windows, outside doors and walls. If outdoors: move away from buildings, trees, streetlights and utility wires and stay in open space. If in a vehicle: stop quickly, stay in the vehicle, avoid stopping near buildings or overpasses. If trapped under debris: do not light a match, tap on a pipe or wall so rescuers can locate you, cover your mouth with a handkerchief.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What are the six pillars of earthquake management according to NDMA guidelines?",
        "ground_truth": "The six pillars are: (1) Earthquake-resistant design and construction of new structures, (2) Selective seismic strengthening and retrofitting of existing priority and lifeline structures, (3) Regulation and enforcement, (4) Awareness and preparedness, (5) Capacity development including education, training, R&D and documentation, and (6) Emergency response.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What percentage of India's land area is vulnerable to moderate or severe earthquakes?",
        "ground_truth": "About 59 percent of India's land area is vulnerable to moderate or severe seismic hazard, prone to shaking of MSK intensity VII and above. During the period 1990 to 2006, India experienced 6 major earthquakes resulting in over 23,000 deaths.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What structures does NDMA prioritize for structural safety audit and seismic retrofitting?",
        "ground_truth": "Priority structures include: buildings of national importance like Parliament House, Supreme Court, Raj Bhavans and heritage buildings; lifeline buildings like schools, colleges, hospitals and tertiary care centres; public utility structures like dams, bridges, ports and airports; governance buildings like district collector offices; and multi-storeyed buildings with five or more floors.",
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