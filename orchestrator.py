from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import the vision tool you built in Phase 2
from vision_tool import analyze_chest_xray

print("Booting Multi-Agent Orchestrator...")

# 1. Initialize the Brain (Llama 3.1)
llm = ChatOllama(
    model="llama3.1",
    temperature=0, 
)

# 2. Re-connect to the Clinical Database (RAG)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vector_store = Chroma(
    persist_directory="./clinical_db",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Convert the RAG Pipeline into a Tool Llama 3.1 can use
rag_tool = create_retriever_tool(
    retriever,
    name="clinical_database_search",
    description="Searches the clinical database for medical literature, standard treatments, and definitions. Always use this to verify clinical facts."
)

# 4. Give the Agent its Toolbox
tools = [analyze_chest_xray, rag_tool]

# 5. Build the Agent Orchestrator using LangGraph
# This sets up the ReAct (Reason + Act) loop automatically
agent = create_react_agent(llm, tools)

print("Orchestrator is online and ready for a case.")

# 6. Test the Orchestrator with a complex clinical request
patient_query = """
I have a chest X-ray image for a patient located at 'data/images/images_normalized/1002_IM-0004-1001.dcm.png'. 
First, look at the image and tell me what you see. 
Then, if you find any abnormalities, search our clinical database to tell me what that specific condition means.
"""

print(f"\nClinician: {patient_query}")
print("Agent working (Watch the reasoning loop)...\n")

# Stream the agent's thought process so we can watch it reason and act
for step in agent.stream({"messages": [("user", patient_query)]}, stream_mode="values"):
    # This will print the LLM's thought process, tool calls, and final answer
    step["messages"][-1].pretty_print()

print("\nFinal Report Complete.")