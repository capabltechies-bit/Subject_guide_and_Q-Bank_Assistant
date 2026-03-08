# 📚 Subject Guide & Question Bank Assistant

> **Capabl.in · AI Agent Development Project — Week 1–2 Milestone**
>
> A multi-document academic assistant that explains topics and solves exam questions using Retrieval-Augmented Generation (RAG).

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-team>/subject-guide-assistant.git
cd subject-guide-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Create a `.env` file (never commit this!):
```
OPENAI_API_KEY=sk-...
# OR
GOOGLE_API_KEY=...
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🏗️ Project Structure

```
subject_guide_assistant/
├── app.py                  # Streamlit UI (4 tabs)
├── document_processor.py   # PDF / DOCX / PPTX / TXT extraction + chunking
├── rag_engine.py           # FAISS vector store + LangChain RAG pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🌟 Features (Week 1–2)

| Feature | Status |
|---|---|
| Multi-format upload (PDF, DOCX, PPTX, TXT) | ✅ |
| Auto content-type detection (textbook / notes / question paper / lab) | ✅ |
| FAISS-powered semantic search | ✅ |
| Topic explanation with structured output | ✅ |
| Exam question solving with step-by-step breakdown | ✅ |
| Source attribution for every answer | ✅ |
| Query history | ✅ |
| OpenAI + Google Gemini support | ✅ |
| Streamlit Cloud deployment | ✅ |

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Add `OPENAI_API_KEY` in **Secrets** (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy** 🎉

---

## 📖 How It Works

```
Upload PDF/DOCX/PPTX
        │
        ▼
document_processor.py
  • Extracts text by format
  • Detects content type via keyword heuristics
  • Splits into overlapping 800-char chunks
        │
        ▼
rag_engine.py  →  AcademicVectorStore
  • Embeds chunks via OpenAI text-embedding-3-small
  • Stores in FAISS IndexFlatL2
        │
        ▼
User Query
  • Embed query → FAISS similarity search → top-k chunks
  • Build prompt with context + question
  • Call GPT-3.5-turbo / Gemini-Pro
  • Return structured answer + source attribution
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35+ |
| Backend | Python 3.11 |
| Orchestration | LangChain 0.2 |
| Vector DB | FAISS (local, in-memory) |
| LLM | OpenAI GPT-3.5-turbo / Google Gemini Pro |
| Embeddings | OpenAI text-embedding-3-small |
| PDF parsing | pdfplumber + PyPDF2 |
| DOCX parsing | python-docx |
| PPTX parsing | python-pptx |

---

## 👥 Team Checklist (Week 1–2)

- [x] GitHub repo with project structure
- [x] Development environment (Python, LangChain, Streamlit, FAISS)
- [x] Multi-format document processing (PDF, DOCX, PPTX)
- [x] Content categorization (notes / textbook / question paper / lab)
- [x] Topic-based retrieval system (FAISS + embeddings)
- [x] Streamlit interface with upload + query UI
- [ ] Deploy on Streamlit Cloud ← **do this as a team**
- [ ] Record 2-minute demo video

---

## 📝 .gitignore template

```
.env
venv/
__pycache__/
*.pyc
*.faiss
*.meta
.streamlit/secrets.toml
```

---

*Built as part of the Capabl.in Subject Guide & QBank AI Agent project.*
