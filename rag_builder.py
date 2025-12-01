import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DATA_PATH = Path("./knowledge_base")
CHROMA_DB_PATH = Path("./chroma_db")
CHROMA_DB_PATH.mkdir(exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_vector_store():
    print("Starting RAG Builder — loading all PDFs & TXT from knowledge_base/")

    documents = []
    supported_exts = {".pdf", ".txt", ".md", ".markdown"}

    for file_path in DATA_PATH.rglob("*.*"):
        if file_path.suffix.lower() not in supported_exts:
            continue

        print(f"Loading → {file_path.name}")

        
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        loaded_docs = loader.load()

        
        for doc in loaded_docs:
            doc.metadata.update({
                "source": file_path.name,
                "title": file_path.stem.replace("_", " ").replace("-", " "),
                "source_folder": str(file_path.parent.name) if file_path.parent != DATA_PATH else "main_knowledge_base",
                "source_type": "book_or_story"
            })
            documents.append(doc)

    if not documents:
        print("No files found! Put your PDFs directly in knowledge_base/")
        return

    print(f"Loaded {len(documents)} files")

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} smart chunks")

    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    
    if os.path.exists(CHROMA_DB_PATH / "chroma.sqlite3"):
        print("Updating existing database...")
        db = Chroma(persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        print("Creating new database...")
        Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DB_PATH))

    print(f"SUCCESS! Your RAG now has {len(chunks)} chunks")
    print(f"Database saved at: {CHROMA_DB_PATH}")
    print("Ready to chat! Run your query script now.")

if __name__ == "__main__":
    build_vector_store()