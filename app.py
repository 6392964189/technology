import os
import faiss
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Load SentenceTransformer model (free and open-source)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS for stylish design
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Helvetica Neue', sans-serif;
        color: #333333;
    }
    .header {
        color: #1f78b4;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #555555;
        text-align: center;
        font-size: 20px;
    }
    .uploaded-text {
        background-color: #e9f7ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .button:hover {
        background-color: #45a049;
    }
    .match-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .match-card h4 {
        color: #1f78b4;
    }
    .match-card p {
        color: #555555;
    }
    </style>
    """, unsafe_allow_html=True
)

# Streamlit interface
st.markdown('<div class="header">PDF Question-Answering System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload your PDF and ask any question!</div>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF content
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    # Split text for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings using SentenceTransformer
    texts = [chunk.page_content for chunk in chunks]
    embeddings = np.array([embedding_model.encode(text) for text in texts])

    # Create FAISS index for storing embeddings
    dimension = embeddings.shape[1]  # Get the embedding dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add embeddings to the FAISS index

    st.success("PDF processed and indexed successfully!")

    st.markdown(f'<div class="uploaded-text">PDF uploaded successfully! Now, ask a question related to its content.</div>', unsafe_allow_html=True)

# Input query
query = st.text_input("Ask a question about the uploaded PDF")
if query:
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query])

    # Perform similarity search
    k = 5  # Top k closest matches
    distances, indices = index.search(query_embedding, k)

    # Display top matches with styled design
    st.markdown('<div class="subheader">Top Matches:</div>', unsafe_allow_html=True)
    for i in range(k):
        st.markdown(f"""
            <div class="match-card">
                <h4>Match {i+1}</h4>
                <p>{texts[indices[0][i]]}</p>
                <p><strong>Score:</strong> {distances[0][i]}</p>
            </div>
        """, unsafe_allow_html=True)
