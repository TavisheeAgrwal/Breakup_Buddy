import streamlit as st
import os
from google import genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets or .env file.")
    st.stop()


CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    st.error("Error: system_prompt.txt not found.")
    st.stop()

@st.cache_resource(ttl=3600) 
def setup_rag_and_client():
    """Sets up the Gemini client, the Chat object, and RAG system once."""
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY) 
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.stop()
        
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    chat = client.chats.create(model="gemini-2.5-flash", config=config)
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        rag_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        st.success("RAG Knowledge Base Loaded Successfully.")
    except Exception as e:
        st.error(f"Error loading RAG system. Did you run 'python rag_builder.py'? Error: {e}")
        st.stop()
        
    return client, chat, rag_retriever

def classify_emotion(client: genai.Client, user_input: str) -> str:

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
            contents=emotion_prompt
        )
        return response.text.strip().replace('[', '').replace(']', '').split(',')[0]
    except Exception:
        return "Neutral"

client, chat_session, rag_retriever = setup_rag_and_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = chat_session 
    st.session_state.messages.append({"role": "model", "content": "Hello! I'm here to listen. Tell me what's on your mind today."})


st.title("💔 Breakup Buddy: Conversational AI Coach")
st.caption("Grounded advice, empathetic persona, and emotional tracking.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to talk about?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Buddy is thinking..."):
        current_emotion = classify_emotion(client, prompt)
        
        retrieved_docs = rag_retriever.invoke(prompt)
        context_chunks = [doc.page_content for doc in retrieved_docs]
        context = "\n--- RELEVANT COPING ADVICE ---\n" + "\n\n".join(context_chunks)

        rag_message_to_send = f"""
        User Query: {prompt}
        
        DETECTED PRIMARY EMOTION: {current_emotion}
        
        CONTEXTUAL KNOWLEDGE FOR AI USE (ONLY IF RELEVANT):
        {context}
        
        Your response must be in character (Breakup Buddy persona), empathetic, and **must prioritize validation and gentle reassurance if the emotion is Sadness, Anger, or Anxiety**. Ground your advice in the provided knowledge if applicable.
        """

        response = st.session_state.chat_session.send_message(rag_message_to_send)

        full_response = response.text
        
    with st.chat_message("model"):
        st.markdown(full_response)

    st.session_state.messages.append({"role": "model", "content": full_response})