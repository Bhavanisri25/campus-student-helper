import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD CAMPUS GUIDELINES
# ============================================================

loader = TextLoader("campus_guidelines.txt")

documents = loader.load()


# ============================================================
# SPLIT DOCUMENT INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

vectorstore = Chroma.from_documents(
    chunks,
    embeddings
)


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate.from_template(
    """
    Answer the student's question using ONLY the context below.

    If the answer isn't in the context,
    say you don't have that information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# ============================================================
# RAG FUNCTION
# ============================================================

def run_agent(query):

    print(f"\n[Observe] Query: {query}")

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    print(f"[Decide] Retrieved {len(docs)} relevant chunk(s)")

    # Create context
    context = "\n\n".join(
        document.page_content
        for document in docs
    )

    # Create chain
    chain = prompt | llm

    # Generate answer
    response = chain.invoke({
        "context": context,
        "question": query
    })

    print(f"[Act] Answer: {response.content}")

    return response.content


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ASK QUESTION API
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    query = data.get("question", "").strip()

    if not query:

        return jsonify({
            "answer": "Please enter a question."
        })


    try:

        answer = run_agent(query)

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "answer": "Sorry, something went wrong."
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )