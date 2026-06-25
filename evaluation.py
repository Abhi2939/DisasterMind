from agents.rag_agent import vector_db
from ragas import evaluate,EvaluationDataset
from ragas.metrics import LLMContextRecall,Faithfulness,FactualCorrectness,LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Test Question Answering 
TEST_SET = [
    # ── CYCLONE (3 questions from IMD SOP) ──────────────────────────────
    {
        "question": "What wind speed range defines a Severe Cyclonic Storm?",
        "ground_truth": "A Severe Cyclonic Storm is defined by maximum sustained winds of 48 to 63 knots (89 to 117 kmph). Above that is Very Severe Cyclonic Storm (64–119 knots, 118–221 kmph) and Super Cyclonic Storm (120 knots and above, ≥222 kmph).",
        "disaster_type": "Cyclone",
    },
    {
        "question": "What are the four stages of IMD's cyclone warning system?",
        "ground_truth": "The four-stage warning system consists of: (1) Pre-Cyclone Watch — issued 72 hours in advance of adverse weather onset, giving early warning of cyclone development; (2) Cyclone Alert — issued at least 48 hours in advance, with location, intensity and advice to fishermen and public; (3) Cyclone Warning — issued at least 24 hours in advance with landfall point, time, storm surge and impact details; (4) Post Landfall Outlook — issued at least 12 hours in advance of expected landfall, covering likely movement and adverse weather in interior areas.",
        "disaster_type": "Cyclone",
    },
    {
        "question": "What colour codes does IMD use for different stages of cyclone warning bulletins?",
        "ground_truth": "IMD uses different colour codes at different warning stages since the post-monsoon season of 2006, as desired by the National Disaster Management authority: Cyclone Alert bulletins use Yellow, Cyclone Warning bulletins use Orange, and Post Landfall Outlook bulletins use Red.",
        "disaster_type": "Cyclone",
    },

    # ── EARTHQUAKE (3 questions from NDMA Guidelines + NDMA Do's & Don'ts) ──
    {
        "question": "What is NDMA's recommended preparedness action for earthquake-prone zones?",
        "ground_truth": "NDMA recommends: repair deep plaster cracks in ceilings and foundations; anchor overhead lighting fixtures; fasten shelves securely to walls; secure water heaters and LPG cylinders by strapping to walls or bolting to floor; identify safe places indoors (under sturdy table, against inside wall) and outdoors (away from buildings, trees, electrical lines); keep a disaster emergency kit with torch, battery radio, first aid kit, dry food, water, medicines and cash; and develop a family emergency communication plan.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "How does NDMA classify earthquake risk zones in India?",
        "ground_truth": "India is divided into four seismic zones per IS:1893 (2002): Zone II (low damage risk, MSK VI or less, 41.40% of land area), Zone III (moderate damage risk, MSK VII, 30.40%), Zone IV (high damage risk, MSK VIII, 17.30%), and Zone V (very high damage risk, MSK IX or more, 10.90%). Zones III, IV and V together cover 58.6% of India's land area and are classified as High Risk Areas, vulnerable to earthquakes, landslides and rock falls.",
        "disaster_type": "Earthquake",
    },
    {
        "question": "What immediate response actions does NDMA recommend during an earthquake?",
        "ground_truth": "If indoors: DROP to the ground, take COVER under a sturdy table or furniture and HOLD ON until shaking stops; stay away from glass, windows and outside walls. If outdoors: move away from buildings, trees, streetlights and utility wires; stay in open space. If in a moving vehicle: stop quickly and stay in the vehicle; avoid stopping near buildings, trees or overpasses. If trapped under debris: do not light a match or move about; tap on a pipe or wall so rescuers can locate you; cover your mouth with a handkerchief.",
        "disaster_type": "Earthquake",
    },
]


def build_dataset():

    rows = []
    for case in TEST_SET:
        results = vector_db.similarity_search(
            case["question"],k=6,filter={"disaster_type":case["disaster_type"]}
        )

        retrieved_chunks = [doc.page_content for doc in results]

        rows.append(
            {
                "user_input":case["question"],
                "retrieved_contexts":retrieved_chunks,
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
        ],
        llm=llm,
        run_config=RunConfig(
            max_workers=2,       
            max_wait=120,
            timeout=60,
        )
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)
    results.to_pandas().to_csv("rag_evaluation_results.csv", index=False)
    print("Saved to rag_evaluation_results.csv")


if __name__ == "__main__":
    main()