"""Flask App script for RAG chatbot"""

import gc
import os
import re
import tempfile
from flask import Flask, request, jsonify, render_template

# # Pre-download and save the embedding model
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
# model.save("models/paraphrase-MiniLM-L3-v2")


# Disable CUDA and excessive parallel threads to save memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Flask app initialization
app = Flask(__name__, template_folder="templates", static_folder="static")


# Global states
retriever = None
LLM_model = None
api_key = None  # API key will come from frontend


SYSTEM_MESSAGE = """
You are a RAG Assistant for the uploaded document.
Your role is to help users understand its contents clearly and accurately.

Rules:
1. Prioritize the document context first.
2. If the answer isn’t in the document, say you don’t know.
3. Be friendly, direct, and concise.
4. Avoid adding extra information unless asked.
"""


# routes
@app.route("/")
def home():
    return render_template("chat_page.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Route handling document upload, splitting, chunking, and vectorization."""
    
    global retriever, LLM_model, api_key

    # Import heavy dependencies only when needed
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = request.form.get("apiKey")
    if not api_key:
        return "API key missing!", 400

    uploaded = request.files.get("file")
    if not uploaded or uploaded.filename.strip() == "":
        return "No file uploaded", 400

    ext = uploaded.filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        uploaded.save(tmp_file.name)
        path = tmp_file.name

    # load document
    try:
        loader = PyPDFLoader(path) if ext == "pdf" else TextLoader(path)
        documents = loader.load()
    except Exception as e:
        os.unlink(path)
        return f"Failed to read document: {e}", 400

    if not documents:
        os.unlink(path)
        return "No readable content found in the document.", 400


    # split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100) # reduce chunk_size for low memory
    chunks = splitter.split_documents(documents)


    # Light embedding model (fast + low memory)
    try:
        embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        # embeds = HuggingFaceEmbeddings(model_name="./models/paraphrase-MiniLM-L3-v2")  # local model (offline)
        vector_store = FAISS.from_documents(chunks, embeds)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
    except Exception as e:
        os.unlink(path)
        return f"Embedding model failed: {e}", 500


    # Initialize chat model
    try:
        LLM_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    except Exception as e:
        return f"Failed to initialize chat model: {e}", 500


    # Cleanup temp file
    os.unlink(path)
    del documents, chunks, vector_store
    gc.collect()

    return "Document processed successfully! You can now ask questions."


@app.route("/chat", methods=["POST"])
def chat():
    """Q&A route on uploaded document."""
    global retriever, LLM_model

    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    if retriever is None or LLM_model is None:
        return jsonify({"error": "Please upload a document first."}), 400

    question = request.form.get("question") or (request.json and request.json.get("question"))
    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Retrieve documents with retriever
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        return jsonify({"error": f"Retriever failed: {e}"}), 500

    # prompt template
    prompt_template = PromptTemplate(
        template=(
            "You are answering strictly based on this document.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # Combine into a pipeline
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        })
        | prompt_template
        | LLM_model
        | StrOutputParser()
    )

    try:
        response = chain.invoke(question).strip()
    except Exception as e:
        response = f"Error generating response: {str(e)}"

    # Clean markdown artifacts
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)

    gc.collect()
    return jsonify({"answer": cleaned})


# run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)