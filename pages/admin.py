import streamlit as st
import os
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from semantic_text_splitter import TextSplitter
from streamlit_autorefresh import st_autorefresh
from tokenizers import Tokenizer
import fitz  # PyMuPDF
from docx import Document
import regex as re
import psutil
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Elasticsearch with authentication
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)

# Initialize SentenceTransformer model
model = SentenceTransformer(os.getenv("MODEL_PATH"))

# Load tokenizer and initialize TextSplitter
tokenizer_path = os.getenv("TOKENIZER_PATH")
tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
max_tokens = int(os.getenv("MAX_TOKENS"))
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

# Admin page title
st.title("🔧 Admin Page")

# Utility Functions
def clean_text(text):
    text = re.sub(r'[^\P{C}]+', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def split_text_semantically(text):
    chunks = splitter.chunks(text)
    return chunks

def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id

def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return clean_text(text)

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type == "text/plain":
        return clean_text(uploaded_file.getvalue().decode("utf-8"))
    elif file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_word(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None

def extract_text_from_word(file):
    doc = Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return clean_text(text)

def index_text(index_name, text, document_name):
    clean_text_content = clean_text(text)
    timestamp = datetime.now().isoformat()
    total_chunks = len(split_text_semantically(clean_text_content))
    progress_text = "Indexing document. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    for i, chunk in enumerate(split_text_semantically(clean_text_content)):
        if chunk:
            doc_id = generate_unique_id(chunk)
            embedding = model.encode(chunk).tolist()
            body = {
                "text": chunk,
                "embedding": embedding,
                "document_name": document_name,
                "timestamp": timestamp
            }
            es.index(index=index_name, id=doc_id, body=body)
            my_bar.progress((i + 1) / total_chunks, text=progress_text)
            time.sleep(0.1)

    my_bar.empty()

# Index Management Functions
def create_index(index_name: str, dims: int):
    field_name = "embedding"
    similarity_type = os.getenv("SIMILARITY_TYPE", "cosine")

    if es.indices.exists(index=index_name):
        st.error("Index already exists")
        return

    mappings = {
        "properties": {
            field_name: {
                "type": "dense_vector",
                "dims": dims,
                "index": "true",
                "similarity": similarity_type,
            }
        }
    }

    es.indices.create(index=index_name, body={"mappings": mappings})
    st.success(f"Index '{index_name}' with {dims} dimensions and similarity '{similarity_type}' created successfully")

def delete_index(index_name: str):
    if not es.indices.exists(index=index_name):
        st.error("Index not found")
        return
    es.indices.delete(index=index_name)
    st.success(f"Index '{index_name}' deleted successfully")

# Sidebar settings
st.sidebar.title("Admin Settings")

# Dropdown for index management actions
index_action = st.sidebar.selectbox("Select Action", ["Create Index", "Delete Index"])

if index_action == "Create Index":
    st.sidebar.subheader("Create a New Index")
    new_index_name = st.sidebar.text_input("New Index Name")
    dims = st.sidebar.number_input("Dimensions (dims)", min_value=1, value=1024)
    if st.sidebar.button("Create Index"):
        create_index(new_index_name, dims)

elif index_action == "Delete Index":
    st.sidebar.subheader("Delete an Index")
    indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
    selected_index_name = st.sidebar.selectbox("Select Index to Delete", options=indexes)
    if st.sidebar.button("Delete Index"):
        delete_index(selected_index_name)

uploaded_file = st.file_uploader("Upload a text, PDF, or Word document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_text = extract_text_from_file(uploaded_file)
    if file_text:
        indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
        index_for_upload = st.selectbox("Select Index to Upload Document", options=indexes)
        if st.button("Index Document"):
            index_text(index_for_upload, file_text, uploaded_file.name)
            st.success(f"Document '{uploaded_file.name}' indexed successfully in '{index_for_upload}'.")

default_display_metrics = os.getenv("DEFAULT_DISPLAY_METRICS", "False").lower() == "true"
display_metrics = st.sidebar.checkbox("Display CPU, Memory, and Model Health", value=default_display_metrics)

def get_model_health():
    try:
        health_url = f"{os.getenv('OPENAI_BASE_URL')}/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            status = response.json().get("status", "unknown")
            return status
        else:
            return "unknown"
    except Exception as e:
        return f"error ({str(e)})"

if display_metrics:
    interval = int(os.getenv("INTERVAL", 5000))
    count = st_autorefresh(interval=interval, key="status_refresh")

    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    model_health = get_model_health()

    row1 = st.columns(3)

    cpu_container = row1[0].container()
    mem_container = row1[1].container()
    health_container = row1[2].container()

    cpu_container.metric(label="CPU Usage", value=f"{cpu_usage}%")
    mem_container.metric(label="Memory Usage", value=f"{mem_usage}%")
    health_container.metric(label="Model Health", value=model_health)
