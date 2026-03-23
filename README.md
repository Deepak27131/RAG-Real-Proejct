# Universal Doc-Intel Pro 🚀

### Advanced RAG-Based AI Chatbot for Documents & Web Intelligence

Universal Doc-Intel Pro is an advanced Retrieval-Augmented Generation (RAG) based AI chatbot built using Streamlit, LangChain, Mistral AI, and ChromaDB.
This system allows users to upload documents, provide web URLs, and interact with them using a conversational AI interface with multi-session chat history and references.

The chatbot intelligently reads PDFs, CSVs, images, documents, and web pages and provides accurate answers based on document context, reducing hallucination and improving reliability.

---

# 🌟 Features

### 🤖 AI Chat System

* Conversational AI chatbot
* Multi-session chat history
* Context-aware responses
* History-aware retriever
* Standalone question reformulation

### 📂 Document Intelligence

* PDF support
* CSV support
* DOCX support
* Image support
* Web URL support
* Large file handling (200MB+ PDF support)

### 🧠 RAG Engine

* Mistral AI Embeddings
* Chroma Vector Database
* LangChain Retrieval Chain
* Context-based answering
* Reference-based responses

### 💬 Smart Chat UI

* Streamlit interface
* Chat history sidebar
* New chat sessions
* Source references
* Clean and professional UI

### 🔐 Secure API Handling

* Environment variable support
* .env file usage
* API key protection

---

# 🏗️ System Architecture

User Input
↓
Streamlit UI
↓
Document Loader (PDF / CSV / Image / URL)
↓
Text Splitter (Chunking)
↓
Mistral Embeddings
↓
Chroma Vector Database
↓
History Aware Retriever
↓
RAG Chain
↓
Mistral LLM
↓
Final AI Response with References

---

# 🧰 Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### AI Models

* Mistral AI (LLM)
* Mistral Embeddings

### Vector Database

* ChromaDB

### Document Processing

* PyPDFLoader
* CSVLoader
* UnstructuredFileLoader
* WebBaseLoader

### Environment

* Python 3.10+
* dotenv

---

# 📁 Project Structure

```
Universal-Doc-Intel-Pro/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│
├── vector_db/
│
└── utils/
```

---

# ⚙️ Installation

### Step 1: Clone Repository

```
git clone https://github.com/your-username/universal-doc-intel-pro.git
cd universal-doc-intel-pro
```

---

### Step 2: Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Linux/Mac

```
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

---

# 🔑 API Key Setup

Create a `.env` file

```
MISTRAL_API_KEY=your_api_key_here
```

---

# ▶️ Run Application

```
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

# 🧪 How It Works

### Step 1

Upload file or enter web URL

### Step 2

Click Build RAG Engine

### Step 3

AI reads document

### Step 4

Ask questions

### Step 5

Get answers with references

---

# 📊 Use Cases

### 📚 Education

Students can ask questions from books and notes

### 🏥 Healthcare

Medical document analysis

### 🏛️ Government

Policy and legal document analysis

### 🏢 Business

Report and data analysis

### 🌐 Research

Web-based knowledge extraction

---

# 🧠 AI Capabilities

Context-based answering
Multi-document understanding
Conversation memory
Reference-based response
Large document processing
Reduced hallucination

---

# 📸 UI Preview

Chat Interface
Sidebar History
Document Upload
RAG Engine

---

# 🔮 Future Improvements

Voice Assistant Integration
Multi-Agent AI System
Offline LLM Support (Ollama)
PDF Highlighting
Knowledge Graph Integration
Database Storage
Authentication System

---

# 🏆 Smart India Hackathon Ready

This project is suitable for:

AI-based Document Intelligence
Knowledge Assistant
Government Data Analyzer
Research Assistant
Education AI System

---

# 👨‍💻 Author

Deepak Prajapati
B.Tech CSE
GenAI and LLM with Adavaced RAG 

---

# 📜 License

MIT License

---

# ⭐ Support

If you like this project, give it a star on GitHub.
