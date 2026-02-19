from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("Initializing Embedding Model...")
# fast and light
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)

print("Creating Mock Clinical Database...")
mock_abstracts = [
    Document(
        page_content="Pneumonia often presents with focal consolidations on chest radiography. Lobar pneumonia is typically caused by Streptococcus pneumoniae.",
        metadata={"source": "PubMed", "id": "12345"}
    ),
    Document(
        page_content="Cardiomegaly on a CXR is defined as a cardiothoracic ratio > 0.5. It is a common finding in congestive heart failure.",
        metadata={"source": "RadiologyKey", "id": "67890"}
    ),
    Document(
        page_content="Tension pneumothorax is a life-threatening condition where air enters the pleural space but cannot escape, causing mediastinal shift.",
        metadata={"source": "Clinical Guidelines", "id": "11121"}
    )
]

print("Building Chroma Vector Database...")
# This creates a persistent local vector store in your project folder
vector_store = Chroma.from_documents(
    documents=mock_abstracts,
    embedding=embeddings,
    persist_directory="./clinical_db"
)

print("Testing Retrieval...")
query = "What is the threshold for an enlarged heart on an X-ray?"
print(f"Query: {query}")

# Fetch the single most mathematically similar document
results = vector_store.similarity_search(query, k=1)

print("\nTop Retrieved Document:")
print(f"Source: {results[0].metadata['source']}")
print(f"Content: {results[0].page_content}")