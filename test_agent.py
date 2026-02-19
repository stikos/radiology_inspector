from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# CHANGE: Switched to phi3.5 for speed
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

print("🏥 Loading Radiology Assistant (Phi-3.5)...")

# Test query
query = "What are the 3 key radiological signs of Pneumonia on a Chest X-ray?"
print(f"❓ Query: {query}")
print("Thinking...")

# Run the model
response = llm.invoke([HumanMessage(content=query)])

print("\n💡 Answer:")
print(response.content)