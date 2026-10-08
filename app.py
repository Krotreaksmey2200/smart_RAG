import os
import re
import time
import unicodedata
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import torch
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Khmer Smart RAG | AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CHATGPT / GEMINI MODERN UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Kantumruy+Pro:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Kantumruy Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top App Bar */
    .chat-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    .chat-brand {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #111827;
    }
    .badge-model {
        background: #F3F4F6;
        color: #374151;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #E5E7EB;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    
    /* Hero Welcome Screen (ChatGPT/Gemini style) */
    .hero-container {
        text-align: center;
        max-width: 680px;
        margin: 2.5rem auto 1.5rem auto;
    }
    .hero-icon {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        line-height: 1.5;
        margin-bottom: 1.8rem;
    }

    /* Direct Evidence Citation Block (Perplexity/Gemini style) */
    .evidence-quote {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 12px 0 8px 0;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #1E293B;
    }
    .evidence-label {
        display: block;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3B82F6;
        margin-bottom: 4px;
    }

    /* Citation Sources Pills */
    .source-pills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
    }
    .source-pill {
        display: inline-flex;
        align-items: center;
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        color: #4B5563;
        transition: all 0.2s;
    }
    .source-pill a {
        color: #2563EB;
        text-decoration: none;
        font-weight: 500;
        margin-left: 3px;
    }
    .source-pill a:hover {
        text-decoration: underline;
    }
    
    /* Clean Sidebar Styling */
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
        margin: 1.2rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. TEXT & SCRAPING UTILITIES ---

def clean_khmer(text):
    """Normalize Unicode & strip invisible zero-width spaces (\\u200b, \\xa0)."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\xa0', ' ')
    text = re.sub(r'\[\d+\]|\[edit\]|\[កែប្រែ\]|\[ត្រូវការអំណះអំណាង\]', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def parse_pasted_urls(text):
    """Extract valid HTTP/HTTPS URLs from raw multiline or comma-separated string."""
    if isinstance(text, list):
        return [str(u).strip() for u in text if str(u).strip().startswith("http")]
    urls = re.findall(r"https?://[^\s,;\"'<>]+", str(text))
    return list(dict.fromkeys(urls))


def scrape_urls(urls, delay=0.1):
    """Scrapes clean article body directly from target Wikipedia URLs."""
    headers = {"User-Agent": "KhmerSmartRAG/ChatGPT-Style-Bot (Research)"}
    documents = []
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            heading = soup.find("h1", {"id": "firstHeading"})
            title = heading.get_text(strip=True) if heading else "Untitled"
            body = soup.find("div", {"id": "mw-content-text"})
            if not body:
                continue
            for noise in body.select("table, sup.reference, span.mw-editsection, style, script, .toc"):
                noise.decompose()
            paras = [p.get_text(strip=True) for p in body.find_all("p") if len(p.get_text(strip=True)) > 25]
            text = "\n\n".join(paras).strip()
            if len(text) > 40:
                documents.append({"title": title, "url": url, "text": text})
            time.sleep(delay)
        except Exception:
            continue
    return pd.DataFrame(documents)


def chunk_khmer(df, chunk_size=350, overlap=50):
    """Sentence-aware chunking preserving Khmer full stop '។'."""
    chunks = []
    chunk_id = 0
    for _, row in df.iterrows():
        title = clean_khmer(str(row.get("title", "")))
        url = str(row.get("url", "")).strip()
        text = clean_khmer(row["text"])
        
        raw_parts = [p.strip() for p in re.split(r'(\n+|។)', text) if p.strip()]
        sentences = []
        for p in raw_parts:
            if p == "។" and sentences:
                sentences[-1] += "។"
            elif p != "។":
                sentences.append(p)
                
        curr = []
        curr_len = 0
        for s in sentences:
            if curr_len + len(s) > chunk_size and curr:
                c_text = " ".join(curr).strip()
                if len(c_text) > 30:
                    chunks.append({"chunk_id": chunk_id, "title": title, "url": url, "text": f"[{title}]: {c_text}"})
                    chunk_id += 1
                curr = [s]
                curr_len = len(s)
            else:
                curr.append(s)
                curr_len += len(s)
        if curr:
            c_text = " ".join(curr).strip()
            if len(c_text) > 30:
                chunks.append({"chunk_id": chunk_id, "title": title, "url": url, "text": f"[{title}]: {c_text}"})
                chunk_id += 1
    return pd.DataFrame(chunks)


# --- 4. CACHED MODEL LOADERS ---

@st.cache_resource(show_spinner="Loading Embedding Model (MiniLM-L12)...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="Loading Local Qwen 2.5 (0.5B-Instruct)...")
def load_qwen_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    return tokenizer, model


# --- 5. VECTOR DATABASE CLASS ---

class KhmerVectorDB:
    def __init__(self, embed_model):
        self.model = embed_model
        self.index = None
        self.metadata = None

    def build(self, chunks_df):
        self.metadata = chunks_df.copy().reset_index(drop=True)
        clean_texts = [clean_khmer(t) for t in chunks_df["text"].tolist()]
        vectors = self.model.encode(clean_texts, normalize_embeddings=True, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors.astype(np.float32)))
        return self

    def search(self, query, top_k=3):
        if self.index is None or self.metadata is None or len(self.metadata) == 0:
            return []
        clean_q = clean_khmer(query)
        q_vec = self.model.encode([clean_q], normalize_embeddings=True)
        dense_dists, dense_indices = self.index.search(
            np.ascontiguousarray(q_vec.astype(np.float32)), 
            len(self.metadata)
        )
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_indices[0], 1)}
        
        stopwords = {"▁តើ", "តើ", "អ្វី", "ណា", "នៃ", "យ៉ាង", "ដូច", "ដូចម្តេច", "ជា", "ក្នុង", "ដែល", "មាន", "ហើយ", "និង", "ឬ", "?", "។"}
        raw_tokens = [t.lstrip("▁ ").strip() for t in self.model.tokenizer.tokenize(clean_q)]
        q_terms = [t for t in raw_tokens if t not in stopwords and len(t) >= 2]
        
        candidates = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            text = clean_khmer(str(row["text"]))
            title = clean_khmer(str(row["title"]))
            dense_rank = dense_ranks[idx]
            dense_sim = float(1.0 - (dense_dists[0][dense_rank - 1] / 2.0))
            kw_count = sum(1 for t in q_terms if t in text)
            title_count = sum(1 for t in q_terms if t in title)
            rrf_score = (1.0 / (40.0 + dense_rank)) + (kw_count * 0.02) + (title_count * 0.015)
            candidates.append({
                "score": rrf_score,
                "similarity": dense_sim,
                "title": row["title"],
                "url": row["url"],
                "text": row["text"]
            })
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = []
        for rank, c in enumerate(candidates[:top_k], 1):
            results.append({
                "rank": rank,
                "similarity": c["similarity"],
                "score": c["score"],
                "title": c["title"],
                "url": c["url"],
                "text": c["text"]
            })
        return results

    def save(self, index_path="vector_db/khmer_faiss.index", meta_path="vector_db/khmer_meta.csv"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        self.metadata.to_csv(meta_path, index=False, encoding="utf-8")

    def load(self, index_path="vector_db/khmer_faiss.index", meta_path="vector_db/khmer_meta.csv"):
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            self.metadata = pd.read_csv(meta_path)
            return True
        return False


# --- 6. ANSWER GENERATION & FACTUAL GROUNDING ---

def extract_best_evidence_sentence(query, chunk_text):
    """Extracts the highest-scoring factual sentence from the top chunk."""
    stopwords = {"តើ", "អ្វី", "ណា", "នៃ", "យ៉ាង", "ដូចម្តេច", "ជា", "ក្នុង", "ដែល", "មាន", "ហើយ", "និង", "ឬ"}
    q_words = [w for w in re.findall(r'[\u1780-\u17FF]+', query) if w not in stopwords and len(w) >= 2]
    raw_sents = [s.strip() for s in re.split(r'(\n+|។)', chunk_text) if s.strip() and s.strip() != "។"]
    sents = []
    for s in raw_sents:
        clean_s = s
        if clean_s.startswith("[") and "]:" in clean_s:
            clean_s = clean_s.split("]:", 1)[1].strip()
        if len(clean_s) > 12:
            sents.append(clean_s if clean_s.endswith("។") else clean_s + "។")
    if not sents:
        return chunk_text
    scored = [(sum(1 for w in q_words if w in s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else sents[0]


def generate_qwen_answer(tokenizer, model, query, context, best_evidence):
    """Generates concise answer using local Qwen 2.5 with factual fallback."""
    prompt_text = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Based strictly on the context above, answer the question accurately in Khmer:"
    )
    messages = [
        {"role": "system", "content": "You are a factual Khmer assistant. Answer accurately in Khmer using only the provided context."},
        {"role": "user", "content": prompt_text}
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = outputs[0][inputs['input_ids'].shape[1]:]
        raw_ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
    except Exception:
        raw_ans = ""
    
    clean_q = re.sub(r"[\s\?។!.,\-_តើ]", "", query)
    clean_a = re.sub(r"[\s\?។!.,\-_តើ]", "", raw_ans)
    khmer_ratio = len(re.findall(r'[\u1780-\u17FF]', raw_ans)) / max(len(raw_ans), 1)
    
    # Ground in authentic Wikipedia evidence if hallucinating or echoing
    if not raw_ans or (clean_a in clean_q) or (clean_q in clean_a) or len(clean_a) < 4 or khmer_ratio < 0.35:
        return best_evidence
    return raw_ans


def generate_gemini_answer(api_key, model_choice, query, context):
    """Generates fluent, accurate Khmer answer using Google Gemini Cloud API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)
    prompt = f"""You are Khmer Smart RAG, an accurate AI assistant specialized in the Khmer language.
Answer the following user question in Khmer directly, clearly, and concisely, based strictly and ONLY on the provided context passages.
Never fabricate facts. If the information is not in the context, reply: 'សូមអភ័យទោស ខ្ញុំរកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារយោងដែលមានទេ' (No relevant information in reference).

Context:
{context}

Question: {query}

Answer in Khmer:"""
    response = model.generate_content(prompt)
    return response.text.strip()


# --- 7. INITIALIZE SESSION STATE ---

embed_model = load_embed_model()

if "vdb" not in st.session_state:
    vdb = KhmerVectorDB(embed_model)
    vdb.load()
    st.session_state.vdb = vdb

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# --- 8. CHATGPT / GEMINI STYLE SIDEBAR ---

with st.sidebar:
    # New Chat Button (ChatGPT style)
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Model Provider</div>', unsafe_allow_html=True)
    provider_choice = st.radio(
        "Select Model Engine:",
        ["🤖 Qwen 2.5 (Local CPU)", "✨ Google Gemini (Cloud API)"],
        index=0,
        label_visibility="collapsed"
    )
    
    gemini_key = ""
    gemini_model = "gemini-1.5-flash"
    
    if "Gemini" in provider_choice:
        gemini_model = st.selectbox(
            "Gemini Model:",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
        env_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        gemini_key = st.text_input(
            "Gemini API Key:",
            value=env_key,
            type="password",
            placeholder="Paste your API key here...",
            help="Get your free API key at https://aistudio.google.com"
        )
        if gemini_key:
            st.caption("🟢 API Key configured")
        else:
            st.caption("⚠️ API key required for Gemini")
    else:
        st.caption("⚡ **Qwen 2.5 (0.5B-Instruct)** runs locally on your Mac CPU. 100% Free & Offline.")

    st.markdown('<div class="sidebar-section-title">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Retrieved Context Chunks:", min_value=1, max_value=5, value=3)

    st.markdown('<div class="sidebar-section-title">Knowledge Base Status</div>', unsafe_allow_html=True)
    curr_vectors = st.session_state.vdb.index.ntotal if st.session_state.vdb.index else 0
    st.metric(label="FAISS Vectors Indexed", value=f"{curr_vectors:,}")
    if st.session_state.vdb.metadata is not None:
        num_docs = st.session_state.vdb.metadata["title"].nunique()
        st.caption(f"📚 {num_docs} Wikipedia articles indexed")

    # Knowledge Base Management inside clean expander
    with st.expander("⚙️ Manage Wikipedia Links", expanded=False):
        st.markdown("**Add / Update Wikipedia URLs:**")
        default_urls = """https://km.wikipedia.org/wiki/កម្ពុជា
https://km.wikipedia.org/wiki/ភ្នំពេញ
https://km.wikipedia.org/wiki/អង្គរវត្ត
https://km.wikipedia.org/wiki/ប្រវត្តិសាស្ត្រកម្ពុជា
https://km.wikipedia.org/wiki/ទន្លេមេគង្គ
https://km.wikipedia.org/wiki/ព្រះបាទជ័យវរ្ម័នទី៧
https://km.wikipedia.org/wiki/បញ្ញាសិប្បនិម្មិត
https://km.wikipedia.org/wiki/ភាសាខ្មែរ
https://km.wikipedia.org/wiki/វប្បធម៌កម្ពុជា
https://km.wikipedia.org/wiki/ប្រាសាទបាយ័ន"""
        kb_text = st.text_area("URLs (One per line):", value=default_urls, height=180)
        if st.button("🔄 Scrape & Rebuild FAISS", use_container_width=True):
            urls = parse_pasted_urls(kb_text)
            if not urls:
                st.error("Please provide at least one valid Wikipedia URL.")
            else:
                with st.spinner("Scraping and rebuilding index..."):
                    docs = scrape_urls(urls)
                    if not docs.empty:
                        chunks = chunk_khmer(docs, chunk_size=350, overlap=50)
                        st.session_state.vdb.build(chunks)
                        st.session_state.vdb.save()
                        st.success(f"Indexed {len(chunks)} chunks!")
                        time.sleep(1)
                        st.rerun()


# --- 9. MAIN CHAT INTERFACE (CHATGPT / GEMINI VIBE) ---

active_model_badge = "Qwen 2.5 (Local)" if "Qwen" in provider_choice else f"Gemini ({gemini_model})"

# Top Bar
st.markdown(f"""
<div class="chat-topbar">
    <div class="chat-brand">
        <span>🇰🇭</span> Khmer Smart RAG
    </div>
    <div>
        <span class="badge-model">⚡ {active_model_badge}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Hero Greeting if no chat history yet (just like ChatGPT & Gemini homepages)
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-icon">✨</div>
        <div class="hero-title">How can I help you today?</div>
        <div class="hero-subtitle">
            Ask questions about Cambodian history, culture, geography, or science.<br>
            Answers are grounded directly on Khmer Wikipedia with verified evidence and citations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prompt starter suggestion cards
    st.markdown("<p style='font-size:0.85rem; color:#6B7280; font-weight:600; margin-bottom:8px;'>SUGGESTED QUESTIONS:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏛️ តើប្រាសាទអង្គរវត្តកសាងឡើងក្នុងរាជ្យស្ដេចអង្គណា?", use_container_width=True):
            st.session_state.pending_prompt = "តើប្រាសាទអង្គរវត្តកសាងឡើងក្នុងរាជ្យស្ដេចអង្គណា?"
            st.rerun()
        if st.button("🏙️ តើទីក្រុងភ្នំពេញជារដ្ឋធានីនៃប្រទេសណា?", use_container_width=True):
            st.session_state.pending_prompt = "តើទីក្រុងភ្នំពេញជារដ្ឋធានីនៃប្រទេសណា?"
            st.rerun()
    with col2:
        if st.button("🇰🇭 តើប្រទេសកម្ពុជាមានឈ្មោះផ្លូវការយ៉ាងដូចម្តេច?", use_container_width=True):
            st.session_state.pending_prompt = "តើប្រទេសកម្ពុជាមានឈ្មោះផ្លូវការយ៉ាងដូចម្តេច?"
            st.rerun()
        if st.button("🤖 តើបញ្ញាសិប្បនិម្មិតសំដៅលើអ្វី?", use_container_width=True):
            st.session_state.pending_prompt = "តើបញ្ញាសិប្បនិម្មិតសំដៅលើអ្វី?"
            st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else ("✨" if "Gemini" in provider_choice else "🤖")
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        
        # Direct Evidence Card
        if msg.get("evidence"):
            st.markdown(f"""
            <div class="evidence-quote">
                <span class="evidence-label">📖 Direct Wikipedia Evidence</span>
                "{msg['evidence']}"
            </div>
            """, unsafe_allow_html=True)
            
        # Sources Citations Pills
        if msg.get("sources"):
            pills = "".join([
                f'<span class="source-pill">📄 <a href="{s["url"]}" target="_blank">{s["title"]}</a> (Match: {s["similarity"]:.1%})</span>'
                for s in msg["sources"]
            ])
            st.markdown(f'<div class="source-pills-row">{pills}</div>', unsafe_allow_html=True)

# Fixed Bottom Chat Input
chat_input_val = st.chat_input("Ask a question in Khmer or English... (e.g. តើប្រាសាទអង្គរវត្តកសាងនៅពេលណា?)")
user_query = chat_input_val if chat_input_val else st.session_state.pending_prompt

if user_query:
    st.session_state.pending_prompt = None
    
    # 1. Append and render User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "evidence": None,
        "sources": []
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Generate and stream Assistant Message
    assistant_avatar = "✨" if "Gemini" in provider_choice else "🤖"
    with st.chat_message("assistant", avatar=assistant_avatar):
        if curr_vectors == 0:
            err = "⚠️ The vector database is currently empty. Please add Wikipedia links in the sidebar to index articles."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "evidence": None, "sources": []})
        else:
            with st.spinner("Searching knowledge base & synthesizing answer..."):
                results = st.session_state.vdb.search(user_query, top_k=top_k)
                
                if not results:
                    no_info = "Sorry, no relevant information found in the existing knowledge base for this query."
                    st.warning(no_info)
                    st.session_state.messages.append({"role": "assistant", "content": no_info, "evidence": None, "sources": []})
                else:
                    top_chunk = results[0]
                    context = "\n---\n".join([r["text"] for r in results])
                    best_evidence = extract_best_evidence_sentence(user_query, top_chunk["text"])
                    
                    # Generate Answer based on Model Selection
                    if "Gemini" in provider_choice:
                        if not gemini_key:
                            final_answer = "⚠️ Please provide your **Gemini API Key** in the sidebar to use Google Gemini, or switch to **Qwen 2.5 (Local CPU)**."
                        else:
                            try:
                                final_answer = generate_gemini_answer(gemini_key, gemini_model, user_query, context)
                            except Exception as e:
                                final_answer = f"⚠️ Gemini API Error: {e}\n\n**Direct Evidence Answer:**\n{best_evidence}"
                    else:
                        tokenizer, qwen_model = load_qwen_model()
                        final_answer = generate_qwen_answer(tokenizer, qwen_model, user_query, context, best_evidence)
                    
                    # Display Answer Text
                    st.markdown(final_answer)
                    
                    # Display Direct Evidence Quote
                    st.markdown(f"""
                    <div class="evidence-quote">
                        <span class="evidence-label">📖 Direct Wikipedia Evidence</span>
                        "{best_evidence}"
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Sources Pills
                    pills = "".join([
                        f'<span class="source-pill">📄 <a href="{s["url"]}" target="_blank">{s["title"]}</a> (Match: {s["similarity"]:.1%})</span>'
                        for s in results
                    ])
                    st.markdown(f'<div class="source-pills-row">{pills}</div>', unsafe_allow_html=True)
                    
                    # Persist message to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "evidence": best_evidence,
                        "sources": results
                    })
