from langchain_community.vectorstores import Chroma
from src.helper import load_embedding, repo_ingestion

from dotenv import load_dotenv

import os
import sys
import shutil
import subprocess

from flask import Flask, request, jsonify, render_template

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(BASE_DIR, "repo")
DB_DIR = os.path.join(BASE_DIR, "db")


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in the .env file"
    )


# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = load_embedding()


# =========================================================
# GROQ LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7
)


# =========================================================
# GLOBAL RAG VARIABLES
# =========================================================

vectordb = None
qa = None
memory = None


# =========================================================
# CREATE / LOAD RAG CHAIN
# =========================================================

def load_rag_chain():

    global vectordb
    global qa
    global memory

    print("Loading Chroma database...")

    # Make sure DB exists
    if not os.path.exists(DB_DIR):

        print("Database does not exist yet.")

        vectordb = None
        qa = None

        return

    # Load Chroma database
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    print("Chroma database loaded.")

    # Create new conversation memory
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Create RAG chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 8
            }
        ),
        memory=memory
    )

    print("RAG chain loaded successfully.")


# =========================================================
# LOAD EXISTING DATABASE WHEN APP STARTS
# =========================================================

load_rag_chain()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def index():

    return render_template("index.html")


# =========================================================
# LOAD GITHUB REPOSITORY
# =========================================================

@app.route("/chatbot", methods=["POST"])
def gitRepo():

    global qa

    try:

        user_input = request.form.get(
            "question",
            ""
        ).strip()

        if not user_input:

            return jsonify({
                "success": False,
                "error": "Please enter a GitHub repository URL."
            }), 400


        print("=" * 60)
        print("Repository URL:", user_input)
        print("=" * 60)


        # -------------------------------------------------
        # Remove old database
        # -------------------------------------------------

        if os.path.exists(DB_DIR):

            print("Removing old database...")

            shutil.rmtree(
                DB_DIR,
                ignore_errors=True
            )


        # -------------------------------------------------
        # Download / extract repository
        # -------------------------------------------------

        print("Downloading repository...")

        repo_ingestion(user_input)


        # -------------------------------------------------
        # Create new vector database
        # -------------------------------------------------

        print("Creating vector database...")

        result = subprocess.run(
            [
                sys.executable,
                "store_index.py"
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )


        # -------------------------------------------------
        # Check store_index.py result
        # -------------------------------------------------

        if result.returncode != 0:

            print("STORE INDEX ERROR:")
            print(result.stdout)
            print(result.stderr)

            return jsonify({
                "success": False,
                "error": (
                    "Error while creating vector database.\n\n"
                    + result.stderr
                )
            }), 500


        print("Vector database created successfully.")


        # -------------------------------------------------
        # Reload RAG chain
        # -------------------------------------------------

        print("Reloading RAG chain...")

        load_rag_chain()


        if qa is None:

            return jsonify({
                "success": False,
                "error": "RAG chain could not be loaded."
            }), 500


        print("Repository ready for questions.")


        return jsonify({
            "success": True,
            "response": "Repository loaded successfully."
        })


    except Exception as e:

        print("=" * 60)
        print("REPOSITORY ERROR")
        print(str(e))
        print("=" * 60)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# CHAT
# =========================================================

@app.route("/get", methods=["POST"])
def chat():

    global qa

    try:

        msg = request.form.get(
            "msg",
            ""
        ).strip()


        # -------------------------------------------------
        # Empty message
        # -------------------------------------------------

        if not msg:

            return jsonify({
                "success": False,
                "response": "Please enter a question."
            }), 400


        # -------------------------------------------------
        # Check whether RAG is ready
        # -------------------------------------------------

        if qa is None:

            return jsonify({
                "success": False,
                "response": (
                    "Please upload a GitHub repository first."
                )
            }), 400


        # -------------------------------------------------
        # Ask question
        # -------------------------------------------------

        print("USER:", msg)


        result = qa.invoke({
            "question": msg
        })


        answer = result.get(
            "answer",
            "I couldn't generate an answer."
        )


        print("AI:", answer)


        return jsonify({
            "success": True,
            "response": str(answer)
        })


    except Exception as e:

        print("=" * 60)
        print("CHAT ERROR")
        print(str(e))
        print("=" * 60)

        return jsonify({
            "success": False,
            "response": str(e)
        }), 500


# =========================================================
# CLEAR REPOSITORY + DATABASE
# =========================================================

@app.route("/clear", methods=["POST"])
def clear():

    global vectordb
    global qa
    global memory

    try:

        print("=" * 60)
        print("CLEARING PROJECT DATA")
        print("=" * 60)


        # -------------------------------------------------
        # Delete repository
        # -------------------------------------------------

        if os.path.exists(REPO_DIR):

            print("Deleting repo folder...")

            shutil.rmtree(
                REPO_DIR,
                ignore_errors=True
            )

            print("repo deleted.")


        # -------------------------------------------------
        # Delete Chroma database
        # -------------------------------------------------

        if os.path.exists(DB_DIR):

            print("Deleting db folder...")

            shutil.rmtree(
                DB_DIR,
                ignore_errors=True
            )

            print("db deleted.")


        # -------------------------------------------------
        # Reset RAG objects
        # -------------------------------------------------

        vectordb = None
        qa = None
        memory = None


        print("Repository and database cleared.")


        return jsonify({
            "success": True,
            "message": (
                "Repository and database "
                "cleared successfully."
            )
        })


    except Exception as e:

        print("=" * 60)
        print("CLEAR ERROR")
        print(str(e))
        print("=" * 60)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )

