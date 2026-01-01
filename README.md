# 🔒 Private & Secure Technical Simplifier

> **A 100% local RAG system that translates complex technical documentation into simple, executive-level language without your data ever leaving your machine.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/Stack-LangChain_Modern-green)
![Ollama](https://img.shields.io/badge/Model-Llama_3.2-orange)
![Privacy](https://img.shields.io/badge/Privacy-Local_Only-red)

## 📖 Project Overview

This project addresses two critical pain points in modern business:
1.  **The Communication Gap:** Bridging the divide between technical engineers and non-technical stakeholders.
2.  **Data Privacy:** Using Large Language Models (LLMs) on sensitive/proprietary documents without uploading them to the cloud.

It uses **Retrieval-Augmented Generation (RAG)** to ingest PDF and DOCX files, retrieve relevant context, and rewrite it using a specific "Simplifier Persona".
[📄 Read Full Documentation](docs/Technical_Documentation.pdf)
## 📚 References
* **[Small Language Models are the Future of Agentic AI](YOUR_LINK_HERE)**
    * *This article explains why SLMs are becoming the standard for private, efficient AI.*
## ⚡ Key Features & Optimization

* **Privacy First:** Built on **Ollama** and **ChromaDB** to ensure all embeddings and processing happen locally. No API keys, no cloud uploads.
* **Optimized Performance:** Originally designed for Llama 3 8B, this project has been **optimized for Llama 3.2 (3B)**. This reduces memory overhead by ~60% and enables smooth real-time token streaming on standard consumer laptops.
* **Modern Stack:** Built using the latest **LangChain** integrations (`langchain-ollama`, `langchain-chroma`) and managed via **uv** for dependency hygiene.
* **Real-Time Streaming:** Responses are streamed token-by-token for an immediate, interactive user experience.

## ⚙️ Prerequisites

* **Python 3.10+**
* **Ollama:** You must have [Ollama installed](https://ollama.com/) and running.
* **Model:** Pull the optimized model (Llama 3.2):
    ```bash
    ollama pull llama3.2
    ```

## 🚀 Installation

This project uses **uv** for fast, clean dependency management.

### Option 1: The Modern Way (Recommended)

1.  **Install uv** (if not installed):
    ```bash
    pip install uv
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install langchain-ollama langchain-chroma langchain-text-splitters langchain-community pypdf docx2txt python-dotenv
    ```

### Option 2: The Classic Way (pip)
If you prefer standard pip:
1.  Create a virtual environment: `python -m venv venv`
2.  Activate it.
3.  Install dependencies: `pip install -r requirements.txt`

## 🔧 Configuration

1.  Create a `.env` file in the root directory.
2.  Add the following configuration (optimized for Llama 3.2):

```ini
# Base URL for local Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optimized model for speed/quality balance
LLM_MODEL=llama3.2

# Local path for the vector database
CHROMA_PATH=chroma_db


🏃 Usage
1- Place Documents: Drop your technical PDF or DOCX files into the data/ folder. (Note: Remove any example files and replace them with your own).

2- Run the CLI:
```bash
python rag_cli.py
```
3- Interact: The system will ingest your documents (creating the DB on the first run) and ask for a query.

Example Interaction
Query: "Summarize the BFT consensus vulnerability in simple terms."

Simplified Answer: "A critical risk was found with the way the system checks for agreement among different nodes... To fix this, the current implementation needs to be replaced with a new one called Threshold BFT..."

##📂 Project Structure:
Secure-Technical-Simplifier/
├── rag_cli.py        # Main Application (Modernized LangChain Stack)
├── data/             # Put your PDFs/DOCXs here
├── .env              # Configuration (Model name, paths)
├── .gitignore        # Crucial for privacy (ignores db/ and env)
└── README.md         # Documentation


##🛡️ Privacy Note
To maintain strict security:

The chroma_db/ folder contains your document embeddings. It is ignored by .gitignore to prevent accidental uploads.

The .env file is also ignored.

## 🤝 Contributing
Feel free to submit issues or pull requests. Please ensure you do not upload any proprietary data/ files in your PRs.
