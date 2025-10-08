"""Flask App script for RAG chatbot (using API key from frontend)"""

# import necessary libraries (others will be imported inside routes to save memory)
import gc
import os
import re
import tempfile
from flask import Flask, request, jsonify, render_template


# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")


# Globals states
retriever = None
LLM_model = None
api_key = None  # API key will come from frontend


# set sytem message
SYSTEM_MESSAGE = """
You are RAG Assistant for the provided document. 
Your role is to help users understand and explore the content of uploaded documents.

Rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
4. Go straight to the point and avoid unnecessary information unless told otherwise.
"""


# routes
@app.route("/")
def home():
    return render_template("chat_page.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    global retriever, LLM_model, messages, api_key
    
    # other imports inside route to save memory
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    
    api_key = request.form.get("apiKey")
    if not api_key:
        return "API key missing!", 400

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        file.save(tmp_file.name)
        file_path = tmp_file.name

    # load document file and split
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    if not documents:
        return "No content found in the document", 400


    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)


    # Embeddings and retriever
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L3-v2") # switched to smalleer and faster model embeddings
    vector_store = FAISS.from_documents(chunks, embeds)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


    # Initialize chat model with API key from user
    LLM_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    
    # Clean up memory
    del documents, chunks, vector_store
    gc.collect()

    
    return "Document processed! You can now ask questions."


@app.route("/chat", methods=["POST"])
def chat():
    global retriever, LLM_model
    
    # remaining imports
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import AIMessage, HumanMessage
    

    if retriever is None or LLM_model is None:
        return jsonify({"error": "Please upload a document first."}), 400

    question = request.form.get("question") or (request.json and request.json.get("question"))
    if not question:
        return jsonify({"error": "No question provided"}), 400

    messages.append(HumanMessage(content=question))


    # Retrieve documents with retriever
    retrieved_documents = retriever.invoke(question)
    context_text = "\n\n".join(document.page_content for document in retrieved_documents)

    prompt_template = PromptTemplate(
        template=(
            "You are answering strictly based on this document.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n",
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda documents: "\n\n".join(doc.page_content for doc in documents)),
        "question": RunnablePassthrough(),
    })

    parser = StrOutputParser()
    
    # combine all to one chain
    main_chain = parallel_chain | prompt_template | LLM_model | parser

    try:
        # Get single response (no stream)
        response = main_chain.invoke(question).strip()
    except Exception as e:
        response = f"Error generating response: {str(e)}"


    # Clean up markdown symbols
    cleaned_response = re.sub(r'\*\*(.*?)\*\*', r'\1', response.strip())
    cleaned_response = re.sub(r'\*(.*?)\*', r'\1', cleaned_response)

    
    gc.collect()
    return jsonify({"answer": cleaned_response})



# run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

