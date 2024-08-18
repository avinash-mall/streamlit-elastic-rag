import streamlit as st
from streamlit_autorefresh import st_autorefresh
import os
import psutil
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests
import hashlib
import regex as re
from docx import Document
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from datetime import datetime
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
import pytz

# Set page configuration
st.set_page_config(layout="wide")

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

# Access OpenAI API environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
instruction_prompt = os.getenv("INSTRUCTION_PROMPT")
num_results = int(os.getenv("NUM_RESULTS", 10))  # Default to 10 if not provided

# Load LLM configuration from .env
temperature = float(os.getenv("TEMPERATURE", 0.1))  # Default to 0.1 if not provided
top_p = float(os.getenv("TOP_P", 0.95))  # Default to 0.95 if not provided
frequency_penalty = float(os.getenv("FREQUENCY_PENALTY", 0.0))  # Default to 0.0 if not provided
presence_penalty = float(os.getenv("PRESENCE_PENALTY", 0.0))  # Default to 0.0 if not provided

# Load other configurations
default_dims = int(os.getenv("DEFAULT_DIMS", 1024))  # Default to 768 dimensions if not provided
interval = int(os.getenv("INTERVAL", 5000))  # Refresh interval in milliseconds
default_display_metrics = os.getenv("DEFAULT_DISPLAY_METRICS", "False").lower() == "true"

# Initialize the OpenAI client
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = instruction_prompt

# Index Management Functions
def create_index(index_name: str, dims: int):
    field_name = "embedding"  # Standardize the field name

    if es.indices.exists(index=index_name):
        st.error("Index already exists")
        return

    # Create the index with specific dimensions
    mappings = {
        "properties": {
            field_name: {
                "type": "dense_vector",
                "dims": dims,
                "index": "true",
                "similarity": "cosine",
            }
        }
    }

    es.indices.create(index=index_name, body={"mappings": mappings})
    st.success(f"Index '{index_name}' with {dims} dimensions created successfully")

def delete_index(index_name: str):
    if not es.indices.exists(index=index_name):
        st.error("Index not found")
        return
    es.indices.delete(index=index_name)
    st.success(f"Index '{index_name}' deleted successfully")

# Define the list_indexes function
def list_indexes():
    # Retrieve all indices
    indexes = es.indices.get_alias(index="*")

    # Filter out system indices (those starting with a dot)
    filtered_indexes = [index for index in indexes.keys() if not index.startswith('.')]

    return filtered_indexes

# Title of the app
st.title("💬 RAG Chatbot")

# Function to get model health status
def get_model_health():
    try:
        health_url = f"{OPENAI_BASE_URL}/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            status = response.json().get("status", "unknown")
            return status
        else:
            return "unknown"
    except Exception as e:
        return f"error ({str(e)})"

# Utility Functions
def clean_text(text):
    # Remove non-printable characters
    text = re.sub(r'[^\P{C}]+', '', text)  # \P{C} matches any character except control characters
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def split_text_semantically(text):
    # Split the document into chunks semantically
    chunks = splitter.chunks(text)
    return chunks

def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return clean_text(text)

# Function to extract text from various file types
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

    for chunk in split_text_semantically(clean_text_content):
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

def search_elasticsearch(query):
    query_embedding = model.encode(query).tolist()
    response = es.search(index="_all", body={
        "size": num_results,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    })
    hits = response['hits']['hits']
    return hits

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize variables as empty lists
context_chunks = []
document_names = []
timestamps = []
messages = []
response = []

# Chat input
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            with st.chat_message("assistant") as assistant_message:
                # Retrieve context from Elasticsearch
                hits = search_elasticsearch(prompt)
                if hits:
                    context_chunks = [hit['_source']['text'] for hit in hits]
                    document_names = [hit['_source']['document_name'] for hit in hits]
                    timestamps = [hit['_source']['timestamp'] for hit in hits]
                context = "\n".join(context_chunks)

                # Prepare messages for LLM with structured context
                messages = [{"role": "system",
                             "content": st.session_state.system_instruction}] if st.session_state.system_instruction else []
                if context_chunks:
                    # Adding structured context with document names and timestamps
                    structured_context = ""
                    for chunk, doc_name, timestamp in zip(context_chunks, document_names, timestamps):
                        structured_context += f"Document Name: {doc_name}\nTimestamp: {timestamp}\nContent: {chunk}\n\n"

                    messages.append(
                        {"role": "assistant", "content": f"Here is the context I found:\n\n{structured_context}"})

                messages.append({"role": "user", "content": prompt})

                # Call the LLM API with the context and user query
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    temperature=temperature,  # Loaded from .env
                    top_p=top_p,  # Loaded from .env
                    frequency_penalty=frequency_penalty,  # Loaded from .env
                    presence_penalty=presence_penalty,  # Loaded from .env
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except APIConnectionError as e:
        st.error("The server could not be reached.")
        st.error(f"Details: {e.__cause__}")
    except RateLimitError as e:
        st.error("Rate limit exceeded; please try again later.")
    except APIStatusError as e:
        st.error(f"An error occurred: {e.status_code}")
        st.error(f"Response: {e.response}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Sidebar settings
st.sidebar.title("Settings")

# Toggle for CPU, Memory, and Model Health display
display_metrics = st.sidebar.checkbox("Display CPU, Memory, and Model Health", value=default_display_metrics)

# Display the instruction prompt
st.caption(f"🚀 {instruction_prompt}")

# Fetch the list of indexes for the dropdown
indexes = list_indexes()

# Index management in the sidebar
st.sidebar.title("Manage Indexes")

# Dropdown to select index name
index_name = st.sidebar.selectbox("Select Index", options=indexes)

# Input for creating a new index
new_index_name = st.sidebar.text_input("New Index Name")
dims = st.sidebar.number_input("Dimensions (dims)", min_value=1, value=default_dims)  # Loaded from .env

if st.sidebar.button("Create Index"):
    create_index(new_index_name, dims)

if st.sidebar.button("Delete Index"):
    delete_index(index_name)

# File uploader for text/PDF/Word documents
uploaded_file = st.file_uploader("Upload a text, PDF, or Word document", type=[
    "txt", "pdf", "docx"
])
if uploaded_file is not None:
    file_text = extract_text_from_file(uploaded_file)
    if file_text:
        index_text(index_name, file_text, uploaded_file.name)
        st.success(f"Document '{uploaded_file.name}' indexed successfully.")

# Conditionally display metrics based on the toggle
if display_metrics:
    # Automatically refresh the CPU, Memory, and Health status every 5 seconds
    count = st_autorefresh(interval=interval, key="status_refresh")  # Loaded from .env

    # Update CPU, Memory, and Health status in real-time
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    model_health = get_model_health()

    # Create rows and containers for CPU, Memory, and Health status
    row1 = st.columns(3)

    cpu_container = row1[0].container()
    mem_container = row1[1].container()
    health_container = row1[2].container()

    # Display the metrics inside their respective containers
    cpu_container.metric(label="CPU Usage", value=f"{cpu_usage}%")
    mem_container.metric(label="Memory Usage", value=f"{mem_usage}%")
    health_container.metric(label="Model Health", value=model_health)

# Debugging option
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

if debug_mode:
    st.write("Debugging Information:")
    st.write(f"Instruction Prompt: {st.session_state.system_instruction}")
    st.write(f"User Query: {prompt}")
    st.write("Retrieved Data:")
    for chunk, doc_name, timestamp in zip(context_chunks, document_names, timestamps):
        st.write(f"- **Document Name**: {doc_name}")
        st.write(f"  **Timestamp**: {timestamp}")
        st.write(f"  **Data**: {chunk}")
    st.write("Messages Sent to LLM:")
    st.write(messages)
    st.write("LLM Response:")
    st.write(response)
