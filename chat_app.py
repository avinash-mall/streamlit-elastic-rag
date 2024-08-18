import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
import os
from elasticsearch import Elasticsearch, exceptions, ElasticsearchWarning
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import warnings

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

# Access OpenAI API environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
instruction_prompt = os.getenv("INSTRUCTION_PROMPT")
num_results = int(os.getenv("NUM_RESULTS", 10))  # Default to 10 if not provided
num_candidates = int(os.getenv("NUM_CANDIDATES", 100))
min_score = float(os.getenv("MIN_SCORE", 1.7))

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

# Initialize the OpenAI client
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = instruction_prompt

# Sidebar settings
st.sidebar.title("Settings")

# Dropdown for selecting Elasticsearch index
indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
selected_index = st.sidebar.selectbox("Select Elasticsearch Index", options=indexes)

# Toggle for enabling debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Utility Functions
def search_elasticsearch(query, index_name):
    query_embedding = model.encode(query).tolist()

    try:
        response = es.search(index=index_name, body={
            "query": {
                "knn": {
                    "query_vector": query_embedding,
                    "field": "embedding",
                    "k": num_results,
                    "num_candidates": num_candidates
                }
            },
            "min_score": min_score
        })
        hits = response['hits']['hits']
        print([d['_score'] for d in hits])
        return hits
    except exceptions.RequestError as e:
        print(f"Request Error: {e.info}")
    except exceptions.ConnectionError as e:
        print(f"Connection Error: {e.errors}")
    except Exception as e:
        print(f"General Error: {str(e)}")

# Title of the app
st.title("💬 RAG Chatbot")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            with st.chat_message("assistant") as assistant_message:
                # Retrieve context from Elasticsearch
                hits = search_elasticsearch(prompt, selected_index)
                context_chunks = []
                document_names = []
                timestamps = []
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
                        structured_context += f"Document Name: {doc_name}\nTimestamp: {timestamp}\nContext: {chunk}\n\n"

                    messages.append(
                        {"role": "assistant", "content": f"Here is the context I found:\n\n{structured_context}"})

                messages.append({"role": "user", "content": prompt})

                # Call the LLM API with the context and user query
                # Prepare the parameters for the LLM API call
                params = {
                    "model": st.session_state["openai_model"],
                    "messages": messages,
                    "stream": True
                }

                # Helper function to safely convert environment variables to float
                def safe_float(value):
                    if value is not None and value.lower() != 'none':
                        return float(value)
                    return None

                # Conditionally add parameters from .env if they are not None
                temperature = safe_float(os.getenv("TEMPERATURE"))
                if temperature is not None:
                    params["temperature"] = float(temperature)

                top_p = safe_float(os.getenv("TOP_P"))
                if top_p is not None:
                    params["top_p"] = float(top_p)

                top_k = safe_float(os.getenv("TOP_K"))
                if top_p is not None:
                    params["top_k"] = float(top_k)

                frequency_penalty = safe_float(os.getenv("FREQUENCY_PENALTY"))
                if frequency_penalty is not None:
                    params["frequency_penalty"] = float(frequency_penalty)

                presence_penalty = safe_float(os.getenv("PRESENCE_PENALTY"))
                if presence_penalty is not None:
                    params["presence_penalty"] = float(presence_penalty)

                # Call the LLM API with the context and user query
                stream = client.chat.completions.create(**params)

                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # If debug mode is enabled, display debug information
            if debug_mode:
                st.write("**Debug Information:**")
                st.write("**Instruction Prompt:**", st.session_state.system_instruction)
                st.write("**User Query:**", prompt)
                st.write("**Retrieved Context Chunks:**", context_chunks)
                st.write("**Messages Sent to LLM:**", messages)
                st.write("**LLM Response:**", response)
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
