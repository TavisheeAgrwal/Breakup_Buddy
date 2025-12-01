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

def classify_emotion(client: genai.Client, user_input: str) -> str:
    """Uses the LLM to classify the primary emotion in the user's text."""
    
    emotion_prompt = f"""
    Analyze the following user text and classify the single primary emotion it expresses.
    
    Choose ONLY one of the following labels: 
    [Sadness, Anger, Anxiety, Confusion, Hope, Relief, Loneliness, Neutral]
    
    Output ONLY the label name, with no other text, explanation, or punctuation.
    
    User Text: "{user_input}"
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=emotion_prompt,
        )

        return response.text.strip().replace('[', '').replace(']', '').split(',')[0]
    except Exception:

        return "Neutral"


def summarize_conversation(client: genai.Client, history: list) -> str:
    """Uses the LLM to summarize the conversation history, focusing on key themes."""
    
    if not history:
        return ""
        
    history_text = "\n".join([f"{item['role']}: {item['content']}" for item in history])

    summary_prompt = f"""
    You are the Breakup Buddy AI. Review the following conversation history. 
    Your task is to provide a brief, reflective summary (1-2 sentences). 
    Focus on: 1) The main emotion(s) discussed, and 2) The key piece of advice given or realized.
    
    Conversation History:
    {history_text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=summary_prompt
        )
        return response.text.strip()
    except Exception:
        return "It seems we've covered a lot today. How are you feeling right now?"


CHROMA_DB_PATH = "./chroma_db"

try:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    RAG_RETRIEVER = vector_store.as_retriever(search_kwargs={"k": 2}) 
    print("RAG System Loaded Successfully.")
    
except Exception as e:
    print(f"Error loading RAG system: {e}. Check if 'chroma_db' exists.")
    RAG_RETRIEVER = None 

CONVERSATION_HISTORY = []
MAX_HISTORY_SIZE = 10 

print("--- Breakup Buddy Activated ---")
print("Hello! I'm here to listen. Tell me what's on your mind today.")
print("Type 'exit' to end the session.\n")

while True:

    if len(CONVERSATION_HISTORY) % 8 == 0 and len(CONVERSATION_HISTORY) > 0:
        print("\n--- BREAKUP BUDDY CHECK-IN ---")
        summary = summarize_conversation(client, CONVERSATION_HISTORY)
        print(f"Buddy Reflection: {summary}")
        print("-----------------------------\n")
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("\nTake care. Remember to be gentle with yourself.")
        break

    try:

        current_emotion = classify_emotion(client, user_input)
        print(f"[DEBUG: Detected Emotion: {current_emotion}]") 

        context = ""
        if RAG_RETRIEVER:
            retrieved_docs = RAG_RETRIEVER.invoke(user_input)
            context_chunks = [doc.page_content for doc in retrieved_docs]
            context = "\n--- RELEVANT COPING ADVICE ---\n" + "\n\n".join(context_chunks)
        

        rag_message_to_send = f"""
        User Query: {user_input}
        
        DETECTED PRIMARY EMOTION: {current_emotion}
        
        CONTEXTUAL KNOWLEDGE FOR AI USE (ONLY IF RELEVANT):
        {context}
        
        Your response must be in character (Breakup Buddy persona), empathetic, and **must prioritize validation and gentle reassurance if the emotion is Sadness, Anger, or Anxiety**. Ground your advice in the provided knowledge if applicable.
        """

        response = chat.send_message(rag_message_to_send)
        
        print(f"Breakup Buddy: {response.text}\n")
        CONVERSATION_HISTORY.append({"role": "user", "content": user_input, "emotion": current_emotion})
        CONVERSATION_HISTORY.append({"role": "model", "content": response.text})
        
        if len(CONVERSATION_HISTORY) > MAX_HISTORY_SIZE:
            CONVERSATION_HISTORY = CONVERSATION_HISTORY[2:]
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break