
# Building a RAG Chatbot for AIO Course Materials

This project focuses on building a basic Retrieval Augmented Generation (RAG) program and applying it to a question-answering system for AIO course materials. The system takes a PDF document and a question as input, then provides an answer.

## Project Overview

The core of this project involves integrating Large Language Models (LLMs) with a retrieval mechanism to enhance answer quality. The pipeline includes:
* **File Document:** The source material (a PDF file).
* **Vector Database:** Stores the vectorized content of the PDF.
* **Retriever:** Fetches relevant information from the vector database.
* **LLM (Vicuna):** Generates the answer based on the retrieved context and the user's question.

## Key Steps and Technologies

1.  **RAG Program Construction:**
    * **Library:** The project utilizes the `LangChain` library, specifically designed for building LLM applications.
    * **Installation:** Essential libraries like `transformers`, `bitsandbytes`, `accelerate`, `langchain`, `langchain-chroma`, `langchain_huggingface`, `python-dotenv`, and `pypdf` are installed.
    * **Vector Database Creation:**
        * **PDF Loading:** `PyPDFLoader` is used to load the PDF document.
        * **Embedding Model:** `HuggingFaceEmbeddings` with the "bkai-foundation-models/vietnamese-bi-encoder" model converts text into vectors, improving query accuracy.
        * **Text Splitter:** `SemanticChunker` is employed for meaningful text segmentation based on semantics, rather than fixed lengths. This improves retrieval accuracy. Key parameters include `buffer_size` (grouping sentences), `breakpoint_threshold_type` ("percentile" for semantic difference), `breakpoint_threshold_amount` (e.g., 95% similarity threshold for splitting), and `min_chunk_size` (minimum characters per chunk, e.g., 500).
        * **Vector Database Initialization:** `Chroma` is used to create and store the vectorized chunks, and a retriever is initialized for querying.
    * **Large Language Model (LLM) Initialization:**
        * **Model:** The `Vicuna 7B v1.5` open-source LLM is selected for its performance and good support for Vietnamese.
        * **Configuration:** `BitsAndBytesConfig` is used for 4-bit quantization (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`) to optimize memory usage, especially on Colab.
        * **Pipeline:** The model and tokenizer are integrated into a `text-generation` pipeline.
    * **Running the Program:** The RAG chain combines the vector database, retriever, and LLM to answer questions related to the PDF content.

2.  **User Interface Development:**
    * **Framework:** `Streamlit` is used to build the interactive web application, allowing for rapid deployment.
    * **Environment Setup:** A Conda virtual environment is recommended to manage dependencies (`conda create -n aio-rag python=3.11`).
    * **Session State:** Streamlit's `session_state` is used to cache loaded models (`embeddings`, `llm`, `rag_chain`) to prevent re-loading on user interactions, ensuring a smooth experience.
    * **PDF Processing Function:** A `process_pdf` function handles uploading, loading, semantic chunking, vector database creation, and RAG chain building. It also cleans up temporary files.
    * **Interface Layout:** The Streamlit app includes sections for PDF upload, model loading status, and a question-answering input field.

## How to Use

1.  **Set up Conda Environment:**
    ```bash
    conda create -n aio-rag python=3.11
    conda activate aio-rag
    ```
2.  **Install Dependencies:**
    ```bash
    pip install transformers==4.40.0 langchain==0.1.20 langchainhub==0.1.15 langchain-chroma==0.1.8 langchain_experimental==0.0.61 langchain-community==0.0.38 langchain_huggingface==0.0.3 python-dotenv==1.0.0 pypdf streamlit==1.36.0
    ```
3.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    (Assuming your code is in `app.py`)

Upon launching, the application will first load the embedding model and then the large language model (which might take some time due to its size). Once loaded, you can upload a PDF file, process it, and then ask questions directly to your RAG chatbot.
