# 🎓 AI Teaching Assistant using RAG

An AI-powered Teaching Assistant that answers course-related questions by understanding lecture content from video recordings. The system transcribes lectures, creates a searchable knowledge base using embeddings and vector search, and provides context-aware answers along with the exact lecture timestamp where the concept was discussed.

## 🚀 Features

* 🎤 Automatic lecture transcription using Whisper Small
* ⏱️ Timestamp-aware responses
* 📚 Multi-lecture knowledge base
* 🔍 Semantic search using vector embeddings
* 🤖 Local LLM-powered question answering using Ollama
* ⚡ Retrieval-Augmented Generation (RAG) pipeline
* 🎯 Returns relevant lecture references and timestamps

## 🛠️ Tech Stack

### AI & NLP

* Whisper Small
* Ollama
* Embedding Models

### RAG Components

* FAISS Vector Database
* Semantic Search
* Retrieval-Augmented Generation (RAG)

### Python Libraries

* LangChain
* OpenAI Whisper
* FAISS
* NumPy
* Pandas

## 📂 Project Workflow

Lecture Videos

↓

Audio Extraction (.mp3)

↓

Whisper Small Transcription

↓

Transcript + Timestamps

↓

Text Chunking

↓

Embedding Generation

↓

FAISS Vector Store

↓

Ollama LLM

↓

Question Answering with Sources

## 📌 Example Query

**User Question**

> Where is SEO explained in the course?

**Response**

SEO is discussed in Lecture 6. The instructor explains search engine optimization fundamentals, indexing, ranking factors, and on-page optimization techniques.

**Source**

* Lecture: SEO and Core Web Vitals
* Timestamp: 12:34 – 18:10

## 📊 Key Highlights

* Processed multiple lecture recordings into searchable knowledge chunks.
* Built an end-to-end RAG pipeline combining transcription, embeddings, retrieval, and LLM-based generation.
* Enabled timestamp-based lecture navigation for quick concept discovery.
* Reduced manual effort required to search through hours of educational content.

## 🎯 Use Cases

* Course revision and exam preparation
* Educational content search
* Lecture summarization
* Timestamp-based topic discovery
* Personalized learning assistant

## 🔮 Future Improvements

* Web-based user interface
* Direct video timestamp navigation
* Multi-course support
* Lecture summarization
* Hybrid search (Vector + Keyword Search)
* Performance evaluation dashboard

## 👨‍💻 Author

Developed as an AI-powered educational assistant project demonstrating practical applications of Retrieval-Augmented Generation (RAG), vector databases, semantic search, and local LLM deployment.
