import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please check your .env file.")
    exit()

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit()

try:
    with open("system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print("Error: system_prompt.txt not found. Please create it.")
    exit()

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT
)

chat = client.chats.create(
    model="gemini-2.5-flash", 
    config=config
)

# --- NEW RAG CONFIGURATION (START) ---

CHROMA_DB_PATH = "./chroma_db"

# 1. Initialize the Embeddings Model (MUST match the one used in rag_builder.py)
try:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load the Vector Store (ChromaDB) and expose the retriever
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    # k=2 retrieves the top 2 most relevant chunks from your knowledge base
    RAG_RETRIEVER = vector_store.as_retriever(search_kwargs={"k": 2}) 
    print("RAG System Loaded Successfully.")
    
except Exception as e:
    print(f"Error loading RAG system: {e}. Check if 'chroma_db' exists.")
    RAG_RETRIEVER = None # Fallback if RAG fails

# --- NEW RAG CONFIGURATION (END) ---

print("--- Breakup Buddy Activated ---")
print("Hello! I'm here to listen. Tell me what's on your mind today.")
print("Type 'exit' to end the session.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("\nTake care. Remember to be gentle with yourself.")
        break

    try:
        # 1. Retrieve Relevant Context using RAG
        context = ""
        if RAG_RETRIEVER:
            retrieved_docs = RAG_RETRIEVER.invoke(user_input)
            # Combine the retrieved document text into a single string
            context_chunks = [doc.page_content for doc in retrieved_docs]
            context = "\n--- RELEVANT COPING ADVICE ---\n" + "\n\n".join(context_chunks)
        
        # 2. Construct the RAG-enhanced Message
        # We wrap the user's input with the retrieved context. 
        # The AI is instructed via the SYSTEM_PROMPT to use this context.
        
        rag_message_to_send = f"""
        User Query: {user_input}
        
        CONTEXTUAL KNOWLEDGE FOR AI USE (ONLY IF RELEVANT):
        {context}
        
        Your response must be in character (Breakup Buddy persona), empathetic, and grounded in the provided advice if applicable.
        """

        # 3. Send the RAG-enhanced message to the LLM
        # The 'chat' object automatically preserves the SYSTEM_PROMPT and history.
        response = chat.send_message(rag_message_to_send)
        
        print(f"Breakup Buddy: {response.text}\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break