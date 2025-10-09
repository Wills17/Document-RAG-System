""" Script to chat with a Gemini model about the content of an uploaded file."""

import os
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

# Variables for global use
chunk_size=1000
chunk_overlap=100
model_name= "sentence-transformers/all-MiniLM-L6-v2"

System_Message = """
You are RAG Assistant for the provided document. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""


# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")


# Input file path
file_path = input("\nEnter path to your document (PDF or TXT): ").strip()
if not os.path.exists(file_path):
    print(f"\nThe file path '{file_path}' does not exist. Please check the path and try again.\n")
    raise FileNotFoundError(f"File not found: {file_path}")
    

# Load document
if file_path.endswith(".pdf") or file_path.endswith(".PDF"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path)

documents = loader.load()



# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = splitter.split_documents(documents)


# embeddings + retriever
embeds = HuggingFaceEmbeddings(model_name=model_name)
vector_store = FAISS.from_documents(chunks, embeds)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize chat model
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Initialize chat history
messages = [SystemMessage(content=System_Message)]

print("\n Your document is ready. Ask anything (type 'exit' to quit).\n")


# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    messages.append(HumanMessage(content=user_input))

    # Retrieve relevant documents
    retrieved_document = retriever.invoke(user_input)
    context_text = "\n\n".join(document.page_content for document in retrieved_document)

    prompt_template = PromptTemplate(
        template="""You are answering based on this document:

        {context}

        Question: {question}""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(lambda documents: "\n\n".join(doc.page_content for doc in documents)),
         "question": RunnablePassthrough()}
    )
    
    parse = StrOutputParser()

    #  main chain
    main_chain = parallel_chain | prompt_template | LLM | parse

    # respond to user
    response = ""
    for chunk in main_chain.stream(user_input):
        response += chunk

    print("AI Bot:", response.strip())
    messages.append(AIMessage(content=response.strip()))
