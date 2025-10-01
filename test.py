""" Script to chat with a Gemini model about the content of an uploaded file."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables (GEMINI_API_KEY must be in .env file)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Gemini API Key:", gemini_api_key)


# Initialize the chatbot (Gemini model)
chatbot = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # change to "gemini-1.5-pro-latest" if enabled
    google_api_key=gemini_api_key
)

def process_file(file_path):
    """Read uploaded file and return its content as text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def chat_with_file(file_path):
    """Chat with the model based on file + user input."""
    file_text = process_file(file_path)
    if not file_text:
        print("❌ Could not read file.")
        return

    print("✅ File loaded. You can now chat about its content.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Chat ended.")
            break

        # Combine file content + user question
        prompt = f"File Content:\n{file_text}\n\nUser Question: {user_input}"

        response = chatbot.invoke(prompt)
        print("Bot:", response.content)


if __name__ == "__main__":
    # Example: replace with any text file path you upload
    file_path = "dataset/Stack vs Heap Memory.txt"
    chat_with_file(file_path)
