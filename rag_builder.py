import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Define where your documents are and where the DB should go
DATA_PATH = "./knowledge_base"
CHROMA_DB_PATH = "./chroma_db"

def build_vector_store():
    print("--- Starting RAG Vector Store Builder ---")

    # 2. Load the documents
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith((".txt", ".md")):
            loader = TextLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("No documents found to load. Check your knowledge_base folder.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # 3. Split the documents into smaller, searchable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} text chunks.")

    # 4. Create the Embeddings Model
    # This model converts the text chunks into numerical vectors
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create and Persist the Vector Store (ChromaDB)
    # The store maps the text chunks to their vectors for fast retrieval
    Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Successfully built and saved ChromaDB to {CHROMA_DB_PATH}")

if __name__ == "__main__":
    build_vector_store()