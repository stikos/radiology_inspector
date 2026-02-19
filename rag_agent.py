from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Initializing RAG Components...")

# 1. Load the exact same embedding model used to create the DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 2. Re-connect to your local Chroma vector database
vector_store = Chroma(
    persist_directory="./clinical_db",
    embedding_function=embeddings
)

# Set up the retriever to fetch the top 2 most relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Initialize your reasoning model (Llama 3.1)
llm = ChatOllama(
    model="llama3.1",
    temperature=0, # Keep it at 0 for factual medical outputs
)

# 4. Create the RAG Prompt Template
# This instructs the model to ONLY use the provided context.
template = """
You are a helpful clinical radiology assistant. Answer the question based ONLY on the following context.
If you do not know the answer based on the context, simply state that you do not have enough information.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Helper function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("Building the LCEL Pipeline...")

# 5. Build the RAG Chain using LCEL
# The flow: (Question) -> Retrieve Context -> Format Context & Question -> Prompt -> LLM -> Output String
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Test the Agent
query = "What is the definition of cardiomegaly on a CXR?"
print(f"\nQuery: {query}")
print("Thinking (Retrieving context & generating answer)...")

# Invoke the chain
response = rag_chain.invoke(query)

print("\nAnswer:")
print(response)