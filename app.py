import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

print("--- Breakup Buddy Activated ---")
print("Hello! I'm here to listen. Tell me what's on your mind today.")
print("Type 'exit' to end the session.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("\nTake care. Remember to be gentle with yourself.")
        break

    try:

        response = chat.send_message(user_input)
        print(f"Breakup Buddy: {response.text}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        break