from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import the vision tool you built in Phase 2
from vision_tool import analyze_chest_xray

print("⚙️ Initializing Dynamic Multi-Agent System...")

# 1. Define the Shared State
class ClinicalState(TypedDict):
    patient_query: str
    image_path: str
    vision_findings: str
    literature_context: str
    final_report: str

# 2. Initialize Resources (Local AI Stack)
llm = ChatOllama(model="llama3.1", temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}
)
# vector_store = Chroma(persist_directory="./clinical_db", embedding_function=embeddings)
vector_store = Chroma(persist_directory="./mimic_clinical_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Define the Router Function
def route_query(state: ClinicalState) -> str:
    print("  -> [Router] Deciding workflow path...")
    prompt = f"""
    Analyze the user query. Does the user want you to review a medical image/X-ray?
    Reply ONLY with 'VISION' if yes, or 'TEXT' if no.
    
    Query: {state['patient_query']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    
    if "VISION" in decision:
        print("  -> [Router Decision] Image detected. Routing to Vision Agent.")
        return "vision_node"
    else:
        print("  -> [Router Decision] Text query only. Skipping Vision Agent.")
        return "literature_node"

# 4. Define the Agents (Nodes)
def vision_agent(state: ClinicalState):
    print("  -> [Vision Agent] Analyzing pixel data...")
    findings = analyze_chest_xray.invoke({"image_path": state["image_path"]})
    return {"vision_findings": findings}

def literature_agent(state: ClinicalState):
    print("  -> [Literature Agent] Querying RAG database...")
    # Adapt search: Use vision findings if they exist, otherwise use the original text query
    search_query = state.get("vision_findings") or state["patient_query"]
    docs = retriever.invoke(search_query)
    context = "\n\n".join([d.page_content for d in docs])
    return {"literature_context": context}

def diagnosis_agent(state: ClinicalState):
    print("  -> [Diagnosis Agent] Synthesizing final response...")
    prompt = f"""
    You are an expert diagnostic radiologist. Answer the query based strictly on the provided inputs.
    
    Patient Query: {state['patient_query']}
    Image Findings (if any): {state.get('vision_findings', 'None')}
    Clinical Literature: {state['literature_context']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}

# 5. Build the LangGraph Workflow
workflow = StateGraph(ClinicalState)

workflow.add_node("vision_node", vision_agent)
workflow.add_node("literature_node", literature_agent)
workflow.add_node("diagnosis_node", diagnosis_agent)

# Add the Conditional Edge at the START
workflow.add_conditional_edges(START, route_query)

# Standard sequential edges for the rest of the graph
workflow.add_edge("vision_node", "literature_node")
workflow.add_edge("literature_node", "diagnosis_node")
workflow.add_edge("diagnosis_node", END)

app = workflow.compile()

# 6. Execute Test Cases
if __name__ == "__main__":
    print("\n================ TEST CASE 1: IMAGE + TEXT ================")
    state_1 = {
        "patient_query": "Review this chest X-ray for cardiomegaly and provide a DDx.",
        "image_path": "data/images/images_normalized/1002_IM-0004-1001.dcm.png",
        "vision_findings": "",
        "literature_context": "",
        "final_report": ""
    }
    result_1 = app.invoke(state_1)
    print(f"\nReport 1:\n{result_1['final_report']}\n")

    print("\n================ TEST CASE 2: PURE TEXT ================")
    state_2 = {
        "patient_query": "What are the common causes of a tension pneumothorax according to clinical guidelines?",
        "image_path": "",
        "vision_findings": "",
        "literature_context": "",
        "final_report": ""
    }
    result_2 = app.invoke(state_2)
    print(f"\nReport 2:\n{result_2['final_report']}\n")