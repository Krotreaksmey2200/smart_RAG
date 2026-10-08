# Khmer Smart RAG

A complete, self-contained Retrieval-Augmented Generation (RAG) pipeline designed for Khmer language Wikipedia articles (km.wikipedia.org) powered by Qwen/Qwen2.5-0.5B-Instruct and paraphrase-multilingual-MiniLM-L12-v2.

---

## Project Structure

```text
smart_RAG/
│
├── app.py                   # Interactive Streamlit GUI (Qwen 2.5 + Gemini Selection)
├── khmer_rag.ipynb          # All-in-One Standalone Jupyter Notebook (1 Single File)
├── requirements.txt         # Dependencies list
│
├── vector_db/
│   ├── khmer_faiss.index    # FAISS Vector Index (Dense Embeddings)
│   └── khmer_meta.csv       # Metadata mapping chunks to article titles and URLs
│
└── README.md                # Project documentation
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch GUI Application (Streamlit)
```bash
streamlit run app.py
```
> **Features**:
> - **Model Choice**: Toggle between Qwen 2.5 (Local CPU) and Google Gemini (Cloud API: 1.5-Flash, 2.0-Flash, 1.5-Pro).
> - **Khmer Q&A**: Fast hybrid search with direct evidence and source attribution.
> - **Knowledge Base Manager**: Paste any Khmer Wikipedia URLs to rescrape and rebuild FAISS index directly from the browser.

### 3. Or Run Standalone Notebook
```bash
jupyter notebook khmer_rag.ipynb
```

---

## Pipeline Architecture

```text
   Pasted Links (Custom URLs of any length)
              ↓
   Targeted Web Scraping (Clean HTML extraction with BeautifulSoup)
              ↓
   Sentence-Aware Cleaning & Chunking (Zero-width space removal + "។" sentence preserving)
              ↓
   Hybrid Vector DB (Dense Embeddings + Tokenizer Keyword Boost via Reciprocal Rank Fusion)
              ↓
   Accurate RAG Engine (Qwen 2.5 LLM + Verifiable Direct Evidence & Source Attribution)
```

---

## Key Highlights

1. **100% Standalone (1 File)**: Everything (Scraping, Cleaning, Chunking, FAISS, Hybrid Search, and Qwen 2.5 generation) is self-contained in `khmer_rag.ipynb`.
2. **Multi-Link Pasting**: Easily paste any number of links (1, 5, 10, or 20) separated by newlines or commas. The scraper focuses strictly on those topics to eliminate crawling drift.
3. **Khmer Unicode Precision**: Strips invisible zero-width spaces (`\u200b`) and segments text along complete sentence boundaries using the Khmer full stop (`។`).
4. **Hybrid Retrieval (RRF)**: Combines dense embeddings from `paraphrase-multilingual-MiniLM-L12-v2` with tokenizer entity keywords via Reciprocal Rank Fusion.
5. **Verifiable Citations**: Every answer includes the direct evidence sentence extracted from the source and clickable source URLs.
