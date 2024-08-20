import streamlit as st
import os
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch, ElasticsearchWarning
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
import pytz
import pandas as pd
import warnings
from PIL import Image
import pytesseract
import io
import re
import unicodedata
from langdetect import detect
from bidi.algorithm import get_display

# Supported languages for OCR
language_options = {
    "eng": "English",
    "ara": "Arabic",
    "chi_sim": "Chinese - Simplified",
    "fra": "French",
    "hin": "Hindi",
    "ita": "Italian",
    "jpn": "Japanese"
}

# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

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
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens, trim=False)

# Admin page title
st.title("🔧 Admin Page")


# Utility Functions
def clean_text(text):
    if text is None:
        return ""

    if isinstance(text, list):
        # Extract the 'text' field from each dictionary and join them into a single string
        text = " ".join(item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in text)

    if not isinstance(text, str):
        raise TypeError(f"Expected a string for cleaning, but got {type(text)}")

    # Detect language of the text
    try:
        language = detect(text)
    except:
        language = "unknown"  # fallback if detection fails

    # Normalize Unicode characters (NFC or NFKC)
    text = unicodedata.normalize('NFKC', text)

    # Handle whitespace: replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    if language == "ar":
        # Arabic-specific cleaning: remove unwanted characters or diacritics
        arabic_cleaner = re.compile(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED\uFE70-\uFEFF]')
        text = arabic_cleaner.sub('', text)

    elif language == "en":
        # English-specific cleaning: convert to lowercase (if required)
        text = text.lower()

    # Remove unwanted symbols (keeping Arabic, English letters, and basic punctuation)
    text = re.sub(r'[^\w\s\u0600-\u06FF.,!?;:()\[\]{}\'"-]', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Handle directional issues for mixed Arabic and English text
    if language == "ar" or "arabic" in text:
        text = get_display(text)

    return text



def split_text_semantically(text):
    chunks = splitter.chunks(text)
    return chunks


def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id


# Function to extract text from PDF using multilingual-pdf2text
def extract_text_from_pdf_ocr(file, language):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        page_text = pytesseract.image_to_string(img, lang=language)
        extracted_text += page_text + "\n"

    pdf_document.close()
    return clean_text(extracted_text)


# Function to extract text from PDF using fitz
def extract_text_from_pdf_fitz(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return clean_text(text)


# Function to extract text based on user selection
def extract_text_from_pdf(file, method, language=None):
    if method == "Read Using PDF OCR":
        return extract_text_from_pdf_ocr(file, language)
    elif method == "Read Normal":
        return extract_text_from_pdf_fitz(file)
    else:
        return None


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


def index_text(index_name, text, document_name, total_files, file_number):
    clean_text_content = clean_text(text)
    # Get the current local time
    local_time = datetime.now()
    # Convert local time to UTC
    utc_time = local_time.astimezone(pytz.utc)
    # Use this UTC time for indexing
    timestamp = utc_time.isoformat()
    chunks = split_text_semantically(clean_text_content)
    total_chunks = len(chunks)
    progress_text = f"Indexing document {file_number}/{total_files}. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    for i, chunk in enumerate(chunks):
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

    my_bar.empty()


def list_documents(index_name: str):
    query = {
        "size": 10000,  # Adjust as needed to list more or fewer documents
        "_source": ["document_name", "timestamp"],
        "query": {
            "match_all": {}
        }
    }
    response = es.search(index=index_name, body=query)

    document_data = {}
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        doc_name = source.get("document_name", "Unknown Document")
        timestamp = source.get("timestamp", "No Timestamp")

        if doc_name in document_data:
            document_data[doc_name]["number_of_chunks"] += 1
        else:
            document_data[doc_name] = {
                "document_name": doc_name,
                "number_of_chunks": 1,
                "timestamp": timestamp
            }

    if not document_data:
        return pd.DataFrame(columns=["document_name", "number_of_chunks", "date_time_added"])

    # Convert to DataFrame
    document_df = pd.DataFrame(document_data.values())
    if "timestamp" in document_df.columns:
        document_df["timestamp"] = pd.to_datetime(document_df["timestamp"], errors='coerce').dt.strftime(
            '%Y-%m-%d %H:%M:%S')
    else:
        document_df["timestamp"] = "No Timestamp"

    document_df = document_df.rename(columns={"timestamp": "date_time_added"})

    return document_df


# Index Management Functions
def create_index(index_name: str):
    field_name = "embedding"
    similarity_type = os.getenv("SIMILARITY_TYPE", "cosine")
    default_dims = int(os.getenv("DEFAULT_DIMS", 1024))  # Use the DEFAULT_DIMS from .env

    if es.indices.exists(index=index_name):
        st.error("Index already exists")
        return

    mappings = {
        "properties": {
            field_name: {
                "type": "dense_vector",
                "dims": default_dims,
                "index": "true",
                "similarity": similarity_type,
            }
        }
    }

    es.indices.create(index=index_name, body={"mappings": mappings})
    st.success(f"Index '{index_name}' with {default_dims} dimensions and similarity '{similarity_type}' created successfully")



def delete_index(index_name: str):
    if not es.indices.exists(index=index_name):
        st.error("Index not found")
        return
    es.indices.delete(index=index_name)
    st.success(f"Index '{index_name}' deleted successfully")


# Sidebar settings
st.sidebar.title("Admin Settings")

# Dropdown for index management actions
index_action = st.sidebar.selectbox("Select Action", ["Create Index", "Delete Index", "List Documents"])

if index_action == "Create Index":
    st.sidebar.subheader("Create a New Index")
    new_index_name = st.sidebar.text_input("New Index Name")

    if st.sidebar.button("Create Index"):
        if not new_index_name.strip():
            st.sidebar.error("Index name cannot be empty.")
        elif any(char in new_index_name for char in r'\/:*?"<>|'):
            st.sidebar.error("Index name contains invalid characters.")
        else:
            create_index(new_index_name)
            st.sidebar.success(f"Index '{new_index_name}' created successfully.")


elif index_action == "Delete Index":
    st.sidebar.subheader("Delete an Index")
    indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
    selected_index_name = st.sidebar.selectbox("Select Index to Delete", options=indexes)
    if st.sidebar.button("Delete Index"):
        delete_index(selected_index_name)

elif index_action == "List Documents":
    st.sidebar.subheader("List Documents in an Index")
    indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
    selected_index_name = st.sidebar.selectbox("Select Index to List Documents", options=indexes)
    if st.sidebar.button("List Documents"):
        document_df = list_documents(selected_index_name)
        if not document_df.empty:
            st.write(f"Documents indexed in '{selected_index_name}':")
            st.table(document_df)
        else:
            st.write(f"No documents found in index '{selected_index_name}'.")

# Allow multiple file uploads
uploaded_files = st.file_uploader("Upload text, PDF, or Word documents", type=["txt", "pdf", "docx"],
                                  accept_multiple_files=True)

if uploaded_files:
    indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
    index_for_upload = st.selectbox("Select Index to Upload Document", options=indexes)

    # Show dropdown to choose extraction method for all files
    extraction_method = st.selectbox("Choose extraction method for all PDFs", ["Read Normal", "Read Using PDF OCR"])

    # If OCR is selected, show the multi-select for languages
    if extraction_method == "Read Using PDF OCR":
        selected_languages = st.multiselect(
            "Choose OCR Languages for all PDFs",
            options=["eng", "ara"] + sorted([key for key in language_options.keys() if key not in ["eng", "ara"]]),
            format_func=lambda x: language_options.get(x, x)
        )

        if selected_languages:
            # Combine the selected languages into the format required (e.g., "deu+eng")
            language = "+".join(selected_languages)
            st.write(f"Selected languages: {language}")
        else:
            st.error("Please select at least one language.")
    else:
        language = None  # No language needed for normal reading

    if st.button("Index All Documents"):
        total_files = len(uploaded_files)
        for file_number, uploaded_file in enumerate(uploaded_files, start=1):
            file_type = uploaded_file.type
            progress_text = f"Processing document {file_number}/{total_files}. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            # Step 1: Text Extraction (50% of the progress)
            if file_type == "application/pdf":
                file_text = extract_text_from_pdf(uploaded_file, extraction_method, language)
            else:
                # Handle non-PDF files as before
                file_text = extract_text_from_file(uploaded_file)

            if file_text:
                my_bar.progress(50, text=f"Document {file_number}/{total_files} - Text extracted successfully.")

                # Step 2: Indexing (remaining 50% of the progress)
                index_text(index_for_upload, file_text, uploaded_file.name, total_files=total_files,
                           file_number=file_number)
                my_bar.progress(100, text=f"Document {file_number}/{total_files} - Indexed successfully.")
                st.success(f"Document '{uploaded_file.name}' indexed successfully in '{index_for_upload}'.")

        st.success(f"All documents have been indexed successfully in '{index_for_upload}'.")

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
