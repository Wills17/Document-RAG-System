"""Flask App script for RAG chatbot"""

# import libraries
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings



# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")


# Flask app
app = Flask(__name__)


# Globals to hold state
retriever = None
chat_model = None
messages = []

SYSTEM_MESSAGE = """
You are RAG Assistant for the provided document. 
Your role is to help users understand and explore the content of uploaded documents.

Rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""


# routes
@app.route("/", methods=["GET"])
def index():
    """Simple upload + chat form UI."""
    return render_template_string("""
    <h2>📄 Gemini RAG Chatbot</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <p><b>Upload document (PDF or TXT):</b></p>
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    <br>
    <form action="/chat" method="post">
        <p><b>Ask a question:</b></p>
        <input type="text" name="question" style="width:300px">
        <input type="submit" value="Ask">
    </form>
    <br>
    <p>⚠️ You must upload a document before chatting.</p>
    """)

@app.route("/upload", methods=["POST"])
def upload_file():
    global retriever, chat_model, messages

    if "file" not in request.files:
        return "❌ No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "❌ Empty filename", 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        file.save(tmp_file.name)
        file_path = tmp_file.name


    # Load document
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)


    documents = loader.load()

    if not documents:
        return "❌ No content found in the document", 400


    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # initiate HuggingFace embeddings
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeds)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    # initialize Gemini model
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)


    # Reset messages
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]

    return "Document processed! You can now ask questions."


@app.route("/chat", methods=["POST"])
def chat():
    global retriever, chat_model, messages
    if retriever is None or chat_model is None:
        return "❌ Please upload a document first.", 400

    question = request.form.get("question") or request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    messages.append(HumanMessage(content=question))

    # Retrieve relevant docs
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt_template = PromptTemplate(
        template="""You are answering based on this document:

        {context}

        Question: {question}""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
    )
        
    parser = StrOutputParser()


    main_chain = parallel_chain | prompt_template | chat_model | parser

    response = ""
    for chunk in main_chain.stream(question):
        response += chunk

    messages.append(AIMessage(content=response.strip()))

    return jsonify({"answer": response.strip()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
