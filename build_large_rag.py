from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("📥 1. Downloading Clinical Dataset...")
# -------------------------------------------------------------------
# FOR TODAY: We use an open-source medical dataset filtered for radiology.
# It downloads instantly without a PhysioNet credential.
print("Loading open-source Radiology subset...")
dataset = load_dataset("harishnair04/mtsamples", split="train")

# Filter the dataset to ONLY include Radiology reports
radiology_data = dataset.filter(lambda example: example['medical_specialty'] and 'Radiology' in example['medical_specialty'])

# Extract the transcription text (taking the first 500 reports to save time for this test)
raw_texts = radiology_data['transcription'][:500]

# -------------------------------------------------------------------
# WHEN YOU GET MIMIC-CXR CREDENTIALS, swap the code above with this:
# print("Loading MIMIC-CXR text reports...")
# dataset = load_dataset("physionet/mimic-cxr", name="reports", split="train[:500]", token="YOUR_HUGGINGFACE_TOKEN")
# raw_texts = dataset['text']
# -------------------------------------------------------------------

print(f"✅ Successfully loaded {len(raw_texts)} radiology reports.")

print("🔪 2. Chunking Text for the Vector Database...")
# We must split long reports into smaller chunks so the embedding model 
# can capture specific clinical facts accurately without losing context.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Number of characters per chunk
    chunk_overlap=50,     # Overlap to ensure sentences aren't cut in half
    length_function=len,
)

documents = []
for i, text in enumerate(raw_texts):
    if text: # Ensure text isn't None
        # Split the text
        chunks = text_splitter.split_text(text)
        
        # Convert chunks into LangChain Document objects with metadata
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": "MTSamples-Radiology", "report_id": str(i)}
            )
            documents.append(doc)

print(f"✅ Split reports into {len(documents)} ingestible chunks.")

print("🧠 3. Initializing Embedding Model (CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("💾 4. Building and Persisting ChromaDB...")
# We use a new directory name so we don't overwrite your Phase 1 database
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./mimic_clinical_db"
)

print("\n🎉 Database successfully built! Your agents now have access to hundreds of real clinical reports.")