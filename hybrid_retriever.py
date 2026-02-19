from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

print("1. Initializing Mock Clinical Data...")
# We will use mock data that mixes semantic concepts with highly specific medical acronyms
mock_docs = [
    Document(page_content="Patient has a history of Non-Small Cell Lung Cancer (NSCLC) with an EGFR L858R mutation. Treated with Osimertinib."),
    Document(page_content="The lung carcinoma showed positive margins. The patient is undergoing targeted therapy for a genetic driver mutation."),
    Document(page_content="Patient presents with severe dyspnea and right lower lobe consolidation. Suspected CAP (Community-Acquired Pneumonia)."),
    Document(page_content="X-ray reveals pleural effusion. The patient complains of shortness of breath and chest tightness.")
]

print("2. Building the Sparse Retriever (BM25 - Exact Match)...")
# BM25 is purely keyword-based. It runs entirely in your RAM.
bm25_retriever = BM25Retriever.from_documents(mock_docs)
bm25_retriever.k = 3 # Fetch top 3 matches

print("3. Building the Dense Retriever (Chroma - Semantic Match)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
# Using an in-memory Chroma for this test
vector_store = Chroma.from_documents(documents=mock_docs, embedding=embeddings)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("4. Combining into an Ensemble Retriever...")
# The Ensemble Retriever combines both lists using Reciprocal Rank Fusion (RRF)
# We weight the exact keyword match slightly higher (0.6) than semantic match (0.4)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.6, 0.4]
)



print("5. Initializing the Cross-Encoder Re-ranker...")
# The Cross-Encoder is a heavier, highly accurate model. 
# It takes the combined results from the Ensemble Retriever and assigns a final, strict relevance score.
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=1) # Only return the absolute best 1 result

# We wrap our Ensemble Retriever inside the Compression Retriever
hybrid_pipeline = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

print("\n6. Testing the Hybrid Pipeline...")

# Test Query: Notice how it contains both semantic intent ("targeted therapy") 
# and an exact keyword ("Osimertinib")
query = "What mutation was the patient on Osimertinib targeted therapy treated for?"

print(f"\nQuery: {query}")
print("Working...")

# Run the final pipeline
final_results = hybrid_pipeline.invoke(query)

print("\nTop Retrieved Document (After Re-ranking):")
for doc in final_results:
    print(f"- {doc.page_content}")