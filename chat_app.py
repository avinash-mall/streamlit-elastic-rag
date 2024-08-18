import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
import os
from elasticsearch import Elasticsearch, exceptions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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

# Toggle for enabling debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Utility Functions
def search_elasticsearch(query):
    query_embedding = model.encode(query).tolist()
    similarity_type = os.getenv("SIMILARITY_TYPE", "cosine")

    script_source = {
        "cosine": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        "dot_product": "dotProduct(params.query_vector, 'embedding')",
        "max_inner_product": "maxInnerProduct(params.query_vector, 'embedding') + 1.0"
    }[similarity_type]

    try:
        response = es.search(index="_all", body={
            "size": num_results,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": script_source,
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        })
        hits = response['hits']['hits']
        return hits
    except exceptions.RequestError as e:
        print(f"Request Error: {e.info}")
    except exceptions.ConnectionError as e:
        print(f"Connection Error: {e.info}")
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
                hits = search_elasticsearch(prompt)
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
                        structured_context += f"Document Name: {doc_name}\nTimestamp: {timestamp}\nContent: {chunk}\n\n"

                    messages.append(
                        {"role": "assistant", "content": f"Here is the context I found:\n\n{structured_context}"})

                messages.append({"role": "user", "content": prompt})

                # Call the LLM API with the context and user query
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    temperature=float(os.getenv("TEMPERATURE", 0.1)),  # Loaded from .env
                    top_p=float(os.getenv("TOP_P", 0.95)),  # Loaded from .env
                    frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", 0.0)),  # Loaded from .env
                    presence_penalty=float(os.getenv("PRESENCE_PENALTY", 0.0)),  # Loaded from .env
                    stream=True,
                )
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
