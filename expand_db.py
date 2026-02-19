from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading existing database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vector_store = Chroma(
    persist_directory="./clinical_db", 
    embedding_function=embeddings
)

print("Injecting Differential Diagnosis logic...")
new_docs = [
    Document(
        page_content="Differential Diagnosis for Cardiomegaly: 1. Congestive Heart Failure (look for pleural effusions and Kerley B lines). 2. Valvular Heart Disease (e.g., mitral regurgitation). 3. Pericardial Effusion (enlarged silhouette without pulmonary vascular congestion).",
        metadata={"source": "StatPearls - Cardiomegaly DDx"}
    ),
    Document(
        page_content="Differential Diagnosis for Lung Consolidation / Pneumonia: 1. Bacterial Pneumonia (typically focal/lobar). 2. Viral Pneumonia (often bilateral/interstitial). 3. Pulmonary Edema (bilateral, associated with cardiomegaly).",
        metadata={"source": "RadiologyKey - Consolidations DDx"}
    )
]

vector_store.add_documents(new_docs)
print("Differential Diagnosis rules successfully added to ChromaDB.")