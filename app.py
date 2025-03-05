import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit UI
st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot with Gemini AI")

with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Upload a PDF to start chatting", type="pdf")

# Process PDF
if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # Create vector store using ChromaDB and HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    # Ensure ChromaDB is initialized correctly
    vector_store = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")

    # User input
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        # Retrieve similar chunks
        matches = vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join([match.page_content for match in matches])

        # Call Gemini API for response
        model = genai.GenerativeModel("gemini-2.0-flash")  # Correct model name
        response = model.generate_content(f"Context: {context}\n\nQuestion: {user_question}")

        # Display answer
        st.subheader("Answer:")
        st.write(response.text if hasattr(response, "text") else response.candidates[0].content)

