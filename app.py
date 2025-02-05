# ✅ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ Check & Install Required Libraries
import importlib.util
import sys

def install_if_missing(package, lib_name=None):
    lib_name = lib_name or package
    if importlib.util.find_spec(lib_name) is None:
        print(f"Installing {package}...")
        !pip install {package} --quiet
    else:
        print(f"{package} is already installed.")

# Core libraries
install_if_missing("PyMuPDF", "fitz")  # Alternative to PyPDF2
install_if_missing("sentence-transformers")
install_if_missing("faiss-cpu")
install_if_missing("transformers")
install_if_missing("streamlit")
install_if_missing("plumber")  # Ensure the latest version

print("✅ Setup Complete!")



import os
import fitz  # PyMuPDF for PDF reading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ✅ Path to your Google Drive PDF folder
pdf_folder = "/content/drive/My Drive/MatriMoniBooks"

# ✅ Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# ✅ Load all PDFs
pdf_texts = []
for file in os.listdir(pdf_folder):
    if file.lower().endswith(".pdf"):
        print(f"Processing {file}...")
        pdf_texts.append(extract_text_from_pdf(os.path.join(pdf_folder, file)))

# ✅ Split text into chunks
def split_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

text_chunks = [chunk for text in pdf_texts for chunk in split_text(text)]

# ✅ Create embeddings & FAISS index
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(chunk_embeddings.shape[1])  # Cosine similarity
index.add(chunk_embeddings)

print(f"✅ FAISS index built with {index.ntotal} vectors.")



import streamlit as st
from transformers import pipeline, set_seed

# ✅ Load GPT-2 model
generator = pipeline('text-generation', model='gpt2', tokenizer='gpt2')
set_seed(42)

# ✅ Retrieve relevant chunks
def retrieve_chunks(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    _, indices = index.search(query_embedding, top_k)
    return [text_chunks[idx] for idx in indices[0]]

# ✅ Generate AI response
def generate_response(query):
    retrieved = retrieve_chunks(query)
    prompt = "You are an expert financial advisor. Based on the following context:\n\n"
    for i, chunk in enumerate(retrieved):
        prompt += f"Context {i+1}: {chunk}\n\n"
    prompt += f"User Query: {query}\n\nAnswer:"

    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text'].split("Answer:")[-1].strip()

# ✅ Streamlit Web App
st.title("📊 MatriMoniBooks Financial Advisor")
st.write("Ask a financial question and get AI-generated responses based on your PDF books.")

query = st.text_input("Enter your question:")
if query:
    st.subheader("AI Response:")
    st.write(generate_response(query))

# ✅ Run in Colab: Use `!streamlit run app.py` in a terminal cell
print("✅ Streamlit app is ready! Run `!streamlit run app.py` in Colab terminal.")



