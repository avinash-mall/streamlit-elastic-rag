# Multilingual RAG Chat Application

This project is a Multilingual Retrieval-Augmented Generation (RAG) chat application built with Streamlit, designed to process and interact with Multilingual PDF,Word,Text documents. The application features a semantic search pipeline, document embedding storage in Elasticsearch, and a chat interface for querying and retrieving relevant document segments.

## Features

- **Multilingual Support:** Utilizes multilingual models for semantic search and document embedding as well as text extraction from pdf.
- **Offline Support:** All the models required can be downloaded using provided scripts. Can be used in an offline environment.
- **Document Embedding:** Embeds documents into Elasticsearch using dense vectors for efficient k-nearest neighbor (kNN) search.
- **Streamlit Frontend:** Provides an intuitive chat interface for querying the documents, with responses that include citations. A conversational chatbot which is context aware.
- **Admin Interface:** Separate admin interface for uploading documents and managing settings. Handle scanned documents using OCR.
- **Multiple User Support:** Capable of handling multiple users simultaneously.

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch 8.11 or higher
- Git
- Any openai api server (LLamafile, Ollama, Text-Generation-Webui etc. or OPENAI api)
- tesseract-ocr (Install from https://tesseract-ocr.github.io/tessdoc/Installation.html. Install Additional languages as required.)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/avinash-mall/streamlit-elastic-rag.git
   cd streamlit-elastic-rag
   ```

2. **Install required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the models and tokenizers:**

   Run the provided scripts to download the necessary models and tokenizers:

   ```bash
   python download_model.py
   python download_tokenizer.py
   ```

4. **Configure the environment:**

   Create a `.env` file in the project root and set up the necessary environment variables, including Elasticsearch credentials and model paths. Use the provided `.env` template as a starting point.

5. **Run the application:**

   Launch the Streamlit app:

   ```bash
   streamlit run chat_app.py
   ```

## File Descriptions

- **chat_app.py:** The main application file containing the Streamlit chat interface.
- **download_model.py:** Script for downloading the multilingual model used for document embedding.
- **download_tokenizer.py:** Script for downloading the tokenizer for the multilingual model.
- **pages/admin.py:** Admin interface for document uploads and application settings.
- **models/:** Directory containing the downloaded models and tokenizers.
- **requirements.txt:** List of required Python packages.
- **.env:** Environment variables configuration file.

## Usage

1. **Admin Interface:**
   - Access the admin interface by navigating to the `pages/admin.py` in the Streamlit app.
   - Upload documents and configure settings.

2. **Chat Interface:**
   - Users can query the uploaded documents through the chat interface.
   - The app will return relevant document segments with citations.

3. **Notes:**
   - If you are using OCR for pdf you need to install tesseract-ocr.
   - In .env Once you choose a SIMILARITY_TYPE as any one of dot_product(you have to normalize the chunks), cosine(recommended), max_inner_product(best recommended), l1norm, hamming, l2norm to create an elastic index, if you want to change it, you have to drop the index and re-create and re-index all your data.
   - Only sentence-transformer based embedding/tokenizer models are supported as of now and hugging-face based are work in progress. Adjust DEFAULT_DIMS to the dimensions of your model. multilingual-e5-large uses 1024 dims.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
