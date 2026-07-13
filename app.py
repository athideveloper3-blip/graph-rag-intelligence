"""
╔══════════════════════════════════════════════════════════════════════╗
║          GRAPH RAG — Advanced Semantic Chunking + Reranking          ║
║          Powered by Groq API  |  Deployed with Streamlit             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os, re, uuid, json, math, time, hashlib, textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from groq import Groq

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from streamlit_chat import message as st_message
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    try:
        import fitz
        PDF_BACKEND = "pymupdf"
    except ImportError:
        try:
            from pypdf import PdfReader
            PDF_BACKEND = "pypdf"
        except ImportError:
            try:
                from PyPDF2 import PdfReader as PdfReader2
                PDF_BACKEND = "pypdf2"
            except ImportError:
                PDF_BACKEND = None

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Glass Intelligence CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0e1a;
    --bg-2:      #0f1625;
    --bg-3:      #141d30;
    --panel:     rgba(255,255,255,0.04);
    --panel-2:   rgba(255,255,255,0.07);
    --border:    rgba(255,255,255,0.08);
    --border-2:  rgba(255,255,255,0.13);
    --lead:      #f0f4ff;
    --muted:     #6b7fa3;
    --dim:       #3a4a6b;
    --indigo:    #6366f1;
    --indigo-2:  #818cf8;
    --purple:    #8b5cf6;
    --violet:    rgba(99,102,241,0.15);
    --violet-2:  rgba(99,102,241,0.25);
    --teal:      #10b981;
    --teal-dim:  rgba(16,185,129,0.15);
    --amber:     #f59e0b;
    --amber-dim: rgba(245,158,11,0.12);
    --rose:      #f43f5e;
    --rose-dim:  rgba(244,63,94,0.12);
    --ice:       #67e8f9;
    --ice-dim:   rgba(103,232,249,0.10);
    --r:         8px;
    --r-lg:      12px;
    --r-xl:      16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--lead);
    -webkit-font-smoothing: antialiased;
}
.main .block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: var(--bg-2);
    border-right: 1px solid var(--border);
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sidebar-brand {
    background: linear-gradient(160deg, rgba(99,102,241,0.18) 0%, rgba(139,92,246,0.10) 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 20px 20px;
}
.sidebar-brand .logo-mark {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.1rem; font-weight: 600;
    color: var(--lead); letter-spacing: -0.3px; line-height: 1;
}
.sidebar-brand .logo-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--indigo), var(--purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.sidebar-brand .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: var(--muted);
    letter-spacing: 2px; text-transform: uppercase; margin-top: 6px;
}
.sidebar-brand .version-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--violet); border: 1px solid var(--violet-2);
    color: var(--indigo-2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; padding: 3px 9px;
    border-radius: 100px; letter-spacing: 1px; margin-top: 10px;
}
.version-pill-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }

.sidebar-section {
    padding: 16px 20px 0; border-bottom: 1px solid var(--border); padding-bottom: 16px;
}
.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.sidebar-section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

input, textarea, select,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-3) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: var(--r) !important;
    color: var(--lead) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    outline: none !important;
}

.stSelectbox > div > div {
    background: var(--bg-3) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: var(--r) !important;
    color: var(--lead) !important;
    font-size: 0.82rem !important;
}

.stSlider > div > div > div { background: var(--border-2) !important; }
.stSlider > div > div > div > div { background: var(--indigo) !important; }

.stButton > button {
    background: var(--panel-2);
    color: var(--lead);
    border: 1px solid var(--border-2);
    border-radius: var(--r);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.82rem;
    padding: 8px 18px;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: var(--violet);
    border-color: var(--indigo);
    color: var(--indigo-2);
}
.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, var(--indigo), var(--purple)) !important;
    border-color: transparent !important; color: white !important;
}
button[kind="primary"]:hover { opacity: 0.88 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent; gap: 0;
    border-bottom: 1px solid var(--border); padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none;
    border-bottom: 2px solid transparent;
    color: var(--muted); font-family: 'Inter', sans-serif;
    font-weight: 500; font-size: 0.82rem;
    padding: 10px 20px; margin-bottom: -1px; transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--lead); }
.stTabs [aria-selected="true"] {
    color: var(--indigo-2) !important;
    border-bottom: 2px solid var(--indigo) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

.streamlit-expanderHeader {
    background: var(--bg-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    color: var(--indigo-2) !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
}
.streamlit-expanderContent {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--r) var(--r) !important;
    padding: 16px !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-3) !important;
    border: 1px dashed var(--border-2) !important;
    border-radius: var(--r-lg) !important;
    padding: 16px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--indigo) !important; }

.stCheckbox > label, .stToggle > label {
    font-size: 0.82rem !important; color: var(--lead) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.7rem !important; letter-spacing: 1px; text-transform: uppercase; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { color: var(--lead) !important; font-weight: 600 !important; font-size: 1.6rem !important; }

.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important; overflow: hidden;
}
.stDataFrame thead th {
    background: var(--bg-3) !important; color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important; letter-spacing: 1px; text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody td {
    background: var(--bg-2) !important; color: var(--lead) !important;
    font-size: 0.82rem !important; border-color: var(--border) !important;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background: var(--bg-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--ice) !important;
}

hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: var(--r) !important; font-size: 0.82rem !important;
    border-width: 1px !important; border-style: solid !important;
}
.stSuccess { background: var(--teal-dim) !important; border-color: rgba(16,185,129,0.3) !important; color: var(--teal) !important; }
.stInfo    { background: var(--ice-dim)  !important; border-color: rgba(103,232,249,0.25) !important; color: var(--ice) !important; }
.stWarning { background: var(--amber-dim)!important; border-color: rgba(245,158,11,0.25) !important; color: var(--amber) !important; }
.stError   { background: var(--rose-dim) !important; border-color: rgba(244,63,94,0.25) !important; color: var(--rose) !important; }

div[data-testid="stSpinner"] > div { border-top-color: var(--indigo) !important; }

.hero-bar {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 50%, rgba(10,14,26,0) 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 28px 18px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0;
}
.hero-title { font-size: 1.4rem; font-weight: 600; color: var(--lead); letter-spacing: -0.4px; line-height: 1; }
.hero-title span { background: linear-gradient(90deg, var(--indigo-2), var(--ice)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 5px; }
.hero-metrics { display: flex; gap: 0; border: 1px solid var(--border); border-radius: var(--r-lg); overflow: hidden; }
.hero-metric { padding: 10px 20px; border-right: 1px solid var(--border); text-align: center; background: var(--panel); min-width: 80px; }
.hero-metric:last-child { border-right: none; }
.hero-metric .hm-val { font-size: 1.3rem; font-weight: 600; color: var(--lead); line-height: 1; }
.hero-metric .hm-val.indigo { color: var(--indigo-2); }
.hero-metric .hm-val.purple { color: #a78bfa; }
.hero-metric .hm-val.teal   { color: var(--teal); }
.hero-metric .hm-val.ice    { color: var(--ice); }
.hero-metric .hm-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px; }

.page-header {
    padding: 0 0 20px 0; border-bottom: 1px solid var(--border);
    margin-bottom: 26px; display: flex; align-items: flex-end; justify-content: space-between;
}
.page-title { font-size: 1.5rem; font-weight: 600; color: var(--lead); letter-spacing: -0.4px; line-height: 1; margin: 0; }
.page-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 5px; }

.section-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 14px; margin-top: 8px;
    display: flex; align-items: center; gap: 10px;
}
.section-heading::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.msg-wrap { margin: 8px 0; }
.msg-user {
    background: var(--panel); border: 1px solid var(--border); border-left: 2px solid var(--dim);
    padding: 14px 18px; border-radius: 0 var(--r-lg) var(--r-lg) 0;
    font-size: 0.88rem; line-height: 1.65; color: #c8d3f0;
}
.msg-user::before { content: 'YOU'; font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 2.5px; color: var(--muted); display: block; margin-bottom: 6px; }
.msg-ai {
    background: var(--violet); border: 1px solid var(--violet-2); border-left: 2px solid var(--indigo);
    padding: 14px 18px; border-radius: 0 var(--r-lg) var(--r-lg) 0;
    font-size: 0.88rem; line-height: 1.7; color: #dde4ff;
}
.msg-ai::before { content: 'GRAPH RAG'; font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 2.5px; color: var(--indigo-2); display: block; margin-bottom: 6px; }

.chunk-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 16px 20px; margin-bottom: 8px; position: relative; overflow: hidden; transition: border-color 0.2s; }
.chunk-card:hover { border-color: var(--border-2); }
.chunk-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(180deg, var(--indigo), var(--purple)); }
.chunk-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.chunk-num { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 500; color: var(--indigo-2); background: var(--violet); border: 1px solid var(--violet-2); padding: 2px 8px; border-radius: 4px; letter-spacing: 1px; }
.chunk-doc { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); background: var(--bg-3); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; }
.chunk-score { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--teal); margin-left: auto; }
.chunk-text { font-size: 0.84rem; color: #c8d3f0; line-height: 1.65; border-top: 1px solid var(--border); padding-top: 10px; margin-top: 4px; }

.tag { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; padding: 2px 7px; border-radius: 4px; margin: 2px 2px 2px 0; letter-spacing: 0.5px; }
.tag-entity  { background: var(--ice-dim); border: 1px solid rgba(103,232,249,0.2); color: var(--ice); }
.tag-keyword { background: var(--amber-dim); border: 1px solid rgba(245,158,11,0.2); color: var(--amber); }
.tag-ok      { background: var(--teal-dim); border: 1px solid rgba(16,185,129,0.2); color: var(--teal); }

.score-track { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
.score-track-label { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--muted); min-width: 60px; }
.score-track-bar { flex: 1; height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.score-track-fill { height: 100%; background: linear-gradient(90deg, var(--indigo) 0%, var(--ice) 100%); border-radius: 2px; }
.score-track-val { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--indigo-2); min-width: 42px; text-align: right; }

.doc-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 14px 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; transition: border-color 0.2s; }
.doc-card:hover { border-color: var(--indigo); }
.doc-icon { width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; background: var(--violet); border: 1px solid var(--violet-2); border-radius: 8px; flex-shrink: 0; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; color: var(--indigo-2); font-weight: 500; }
.doc-info { flex: 1; min-width: 0; }
.doc-name { font-weight: 500; font-size: 0.85rem; color: var(--lead); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--muted); margin-top: 3px; }

.graph-ctx { background: var(--bg-3); border: 1px solid var(--border); border-left: 2px solid var(--amber); border-radius: 0 var(--r) var(--r) 0; padding: 14px 18px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #c8d3f0; line-height: 1.7; white-space: pre-wrap; }

.cap-row { display: flex; flex-direction: column; gap: 4px; }
.cap-item { display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); padding: 5px 0; border-bottom: 1px solid var(--border); }
.cap-item:last-child { border-bottom: none; }
.cap-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.cap-dot.on  { background: var(--teal); box-shadow: 0 0 6px rgba(16,185,129,0.5); }
.cap-dot.off { background: var(--amber); box-shadow: 0 0 6px rgba(245,158,11,0.4); }

.sidebar-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 16px 20px; }
.sidebar-stat { background: var(--panel); border: 1px solid var(--border); border-radius: var(--r); padding: 10px 12px; text-align: center; }
.sidebar-stat .v { font-weight: 600; font-size: 1.2rem; color: var(--lead); line-height: 1; }
.sidebar-stat .l { font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px; }
.sidebar-stat.indigo .v { color: var(--indigo-2); }
.sidebar-stat.teal .v { color: var(--teal); }

/* ── Inline setup card ── */
.inline-setup-card {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.inline-setup-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase;
    color: #818cf8; margin-bottom: 14px;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--dim); }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    doc_name: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    semantic_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    graph_context: str
    communities: List[List[str]]
    retrieval_scores: Dict[str, float]
    latency_ms: float


# ═══════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

try:
    import shutil as _shutil
    import pytesseract
    from PIL import Image
    # Only force the Windows path when actually on Windows; on Linux
    # (Streamlit Community Cloud) tesseract-ocr installs to PATH via packages.txt.
    if os.name == "nt":
        _win_tess = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(_win_tess):
            pytesseract.pytesseract.tesseract_cmd = _win_tess
    elif _shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = _shutil.which("tesseract")
    TESSERACT_AVAILABLE = bool(_shutil.which("tesseract")) or os.name == "nt"
except ImportError:
    TESSERACT_AVAILABLE = False

EASYOCR_AVAILABLE = False


class PDFExtractor:

    @staticmethod
    def extract(pdf_bytes: bytes, filename: str = "document.pdf") -> Tuple[str, Dict]:
        if PDF_BACKEND == "pdfplumber":
            return PDFExtractor._pdfplumber(pdf_bytes)
        elif PDF_BACKEND == "pymupdf":
            return PDFExtractor._pymupdf(pdf_bytes)
        elif PDF_BACKEND == "pypdf":
            return PDFExtractor._pypdf(pdf_bytes)
        elif PDF_BACKEND == "pypdf2":
            return PDFExtractor._pypdf2(pdf_bytes)
        else:
            raise RuntimeError("No PDF library found.")

    @staticmethod
    def _pdfplumber(data: bytes) -> Tuple[str, Dict]:
        import io, pdfplumber

        pages_text = []
        meta = {
            "backend": "pdfplumber",
            "ocr_image_regions": 0,
            "ocr_table_regions": 0,
        }

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            meta["pages"] = len(pdf.pages)
            if pdf.metadata:
                meta["title"]  = pdf.metadata.get("Title", "")
                meta["author"] = pdf.metadata.get("Author", "")

            for page in pdf.pages:
                page_parts = []
                page_img_pil = None

                body_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if body_text.strip():
                    page_parts.append(body_text)

                is_fully_scanned = (not body_text.strip() and len(page.images) >= 1)
                if is_fully_scanned and TESSERACT_AVAILABLE:
                    try:
                        page_img_pil = page.to_image(resolution=300).original.convert("RGB")
                        ocr_text = pytesseract.image_to_string(page_img_pil, config='--psm 1').strip()
                        if ocr_text:
                            page_parts.append(ocr_text)
                            meta["ocr_image_regions"] += 1
                    except Exception:
                        pass
                    pages_text.append("\n\n".join(filter(None, page_parts)))
                    continue

                if TESSERACT_AVAILABLE and page.images:
                    try:
                        page_img_pil = page.to_image(resolution=300).original.convert("RGB")
                        scale_x = page_img_pil.width  / float(page.width)
                        scale_y = page_img_pil.height / float(page.height)

                        for img_obj in page.images:
                            x0 = max(0, int(img_obj.get("x0",   0) * scale_x) - 5)
                            y0 = max(0, int(img_obj.get("top",  0) * scale_y) - 5)
                            x1 = min(page_img_pil.width,  int(img_obj.get("x1",     0) * scale_x) + 5)
                            y1 = min(page_img_pil.height, int(img_obj.get("bottom", 0) * scale_y) + 5)

                            if (x1 - x0) < 20 or (y1 - y0) < 20:
                                continue

                            cropped = page_img_pil.crop((x0, y0, x1, y1))
                            ocr_text = pytesseract.image_to_string(cropped, config='--psm 6').strip()
                            if ocr_text:
                                page_parts.append(f"[IMAGE]\n{ocr_text}")
                                meta["ocr_image_regions"] += 1
                    except Exception:
                        pass

                try:
                    tables = page.find_tables()
                    for table in tables:
                        rows = table.extract()
                        if rows and any(any(cell for cell in row) for row in rows):
                            lines = []
                            for row in rows:
                                cells = [str(c or "").replace("\n", " ").strip() for c in row]
                                lines.append(" | ".join(cells))
                            table_text = "\n".join(lines)
                            if table_text.strip():
                                page_parts.append(f"[TABLE]\n{table_text}")
                        elif TESSERACT_AVAILABLE:
                            if page_img_pil is None:
                                page_img_pil = page.to_image(resolution=300).original.convert("RGB")
                                scale_x = page_img_pil.width  / float(page.width)
                                scale_y = page_img_pil.height / float(page.height)
                            x0 = max(0, int(table.bbox[0] * scale_x) - 5)
                            y0 = max(0, int(table.bbox[1] * scale_y) - 5)
                            x1 = min(page_img_pil.width,  int(table.bbox[2] * scale_x) + 5)
                            y1 = min(page_img_pil.height, int(table.bbox[3] * scale_y) + 5)
                            cropped = page_img_pil.crop((x0, y0, x1, y1))
                            ocr_text = pytesseract.image_to_string(cropped, config='--psm 6').strip()
                            if ocr_text:
                                page_parts.append(f"[TABLE-OCR]\n{ocr_text}")
                                meta["ocr_table_regions"] += 1
                except Exception:
                    pass

                pages_text.append("\n\n".join(filter(None, page_parts)))

        full_text = PDFExtractor._clean("\n\n".join(pages_text))
        return full_text, meta

    @staticmethod
    def _page_to_pil(page, resolution: int = 300):
        try:
            pil_img = page.to_image(resolution=resolution).original
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return pil_img
        except Exception:
            return None

    @staticmethod
    def _preprocess_for_ocr(pil_crop):
        from PIL import ImageFilter, ImageEnhance, ImageOps
        import numpy as np

        img = pil_crop.convert("RGB")
        min_width = 1500
        if img.width < min_width:
            scale = min_width / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(2.5)
        img = ImageEnhance.Sharpness(img).enhance(3.0)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        gray = img.convert("L")
        arr  = np.array(gray)
        block_size = 40
        result_arr = np.zeros_like(arr)
        h, w = arr.shape
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = arr[y:y+block_size, x:x+block_size]
                if block.size == 0:
                    continue
                local_mean = block.mean()
                local_std  = block.std()
                thresh = local_mean - local_std * 0.2
                result_arr[y:y+block_size, x:x+block_size] = np.where(
                    block < thresh, 0, 255
                ).astype(np.uint8)

        return Image.fromarray(result_arr).convert("RGB")

    @staticmethod
    def _ocr_region(page_pil, bbox_obj: dict, scale_x: float, scale_y: float, label: str = "") -> str:
        try:
            x0     = float(bbox_obj.get("x0",     0))
            top    = float(bbox_obj.get("top",    0))
            x1     = float(bbox_obj.get("x1",     0))
            bottom = float(bbox_obj.get("bottom", 0))

            px0 = int(x0     * scale_x)
            py0 = int(top    * scale_y)
            px1 = int(x1     * scale_x)
            py1 = int(bottom * scale_y)

            pad = 15
            px0 = max(0, px0 - pad)
            py0 = max(0, py0 - pad)
            px1 = min(page_pil.width,  px1 + pad)
            py1 = min(page_pil.height, py1 + pad)

            if (px1 - px0) < 10 or (py1 - py0) < 10:
                return ""

            cropped = page_pil.crop((px0, py0, px1, py1))

            best_text = PDFExtractor._run_ocr_attempt(cropped, preprocess=True, text_threshold=0.5, low_text=0.3)
            if not best_text:
                best_text = PDFExtractor._run_ocr_attempt(cropped, preprocess=False, text_threshold=0.3, low_text=0.2)
            if not best_text:
                from PIL import ImageOps
                inverted = ImageOps.invert(cropped.convert("RGB"))
                best_text = PDFExtractor._run_ocr_attempt(inverted, preprocess=True, text_threshold=0.3, low_text=0.2)

            if not best_text:
                return ""
            return f"{label}\n{best_text}" if label else best_text

        except Exception:
            return ""

    @staticmethod
    def _run_ocr_attempt(pil_img, preprocess: bool, text_threshold: float, low_text: float) -> str:
        try:
            if preprocess:
                img = PDFExtractor._preprocess_for_ocr(pil_img)
            else:
                img = pil_img.convert("RGB")

            ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            return ocr_text if len(ocr_text) >= 2 else ""
        except Exception:
            return ""

    @staticmethod
    def _format_table(rows: list) -> str:
        lines = []
        for row in rows:
            cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _pymupdf(data: bytes) -> Tuple[str, Dict]:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        meta = {"backend": "pymupdf", "pages": doc.page_count,
                "title": doc.metadata.get("title",""), "author": doc.metadata.get("author","")}
        pages_text = [page.get_text("text") for page in doc]
        doc.close()
        return PDFExtractor._clean("\n\n".join(pages_text)), meta

    @staticmethod
    def _pypdf(data: bytes) -> Tuple[str, Dict]:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        meta = {"backend": "pypdf", "pages": len(reader.pages),
                "title": (reader.metadata or {}).get("/Title",""),
                "author": (reader.metadata or {}).get("/Author","")}
        return PDFExtractor._clean("\n\n".join(p.extract_text() or "" for p in reader.pages)), meta

    @staticmethod
    def _pypdf2(data: bytes) -> Tuple[str, Dict]:
        import io
        from PyPDF2 import PdfReader as PdfReader2
        reader = PdfReader2(io.BytesIO(data))
        meta = {"backend": "PyPDF2", "pages": len(reader.pages), "title":"", "author":""}
        return PDFExtractor._clean("\n\n".join(p.extract_text() or "" for p in reader.pages)), meta

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\x00", "")
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\S\n]{2,}', ' ', text)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC CHUNKER
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedSemanticChunker:
    def __init__(self, embed_model=None, window_size=3, similarity_threshold=0.45,
                 min_chunk_tokens=80, max_chunk_tokens=512, overlap_sentences=1):
        self.embed_model = embed_model
        self.window_size = window_size
        self.sim_threshold = similarity_threshold
        self.min_tokens = min_chunk_tokens
        self.max_tokens = max_chunk_tokens
        self.overlap = overlap_sentences

    @staticmethod
    def _approx_tokens(text): return len(text.split())

    @staticmethod
    def _cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    def _split_sentences(self, text):
        if SPACY_AVAILABLE:
            doc = nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if s.strip()]

    def _embed(self, texts):
        if self.embed_model is not None:
            return self.embed_model.encode(texts, show_progress_bar=False, batch_size=32)
        vocab = {}
        for t in texts:
            for w in t.lower().split():
                if w not in vocab: vocab[w] = len(vocab)
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split(): mat[i, vocab[w]] += 1
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def chunk(self, text, doc_id, doc_name):
        sentences = self._split_sentences(text)
        if not sentences: return []
        embeddings = self._embed(sentences)
        breaks = set()
        for i in range(1, len(sentences)):
            lo = max(0, i - self.window_size); hi = min(len(sentences), i + self.window_size)
            vec_a = embeddings[lo:i].mean(axis=0); vec_b = embeddings[i:hi].mean(axis=0)
            if self._cosine(vec_a, vec_b) < self.sim_threshold: breaks.add(i)
        current_tokens = 0
        for i, s in enumerate(sentences):
            current_tokens += self._approx_tokens(s)
            if current_tokens >= self.max_tokens: breaks.add(i+1); current_tokens = 0
        boundaries = sorted(breaks); segments = []; prev = 0
        for b in boundaries:
            seg = sentences[prev:b]
            if seg: segments.append(seg)
            prev = max(0, b - self.overlap)
        if prev < len(sentences): segments.append(sentences[prev:])
        merged = []
        for seg in segments:
            if merged and self._approx_tokens(" ".join(seg)) < self.min_tokens: merged[-1].extend(seg)
            else: merged.append(seg)
        chunks = []; char_offset = 0
        for idx, seg in enumerate(merged):
            chunk_text = " ".join(seg)
            start = text.find(chunk_text[:40].strip(), char_offset)
            if start == -1: start = char_offset
            end = start + len(chunk_text); char_offset = max(char_offset, end - 50)
            if len(seg) > 1:
                seg_embs = self._embed(seg); sims = []
                for a in range(len(seg_embs)):
                    for b in range(a+1, len(seg_embs)): sims.append(self._cosine(seg_embs[a], seg_embs[b]))
                coherence = float(np.mean(sims)) if sims else 1.0
            else:
                coherence = 1.0
            chunk_emb = self._embed([chunk_text])[0]
            chunks.append(Chunk(
                id=str(uuid.uuid4()), text=chunk_text, doc_id=doc_id, doc_name=doc_name,
                chunk_index=idx, start_char=start, end_char=end, embedding=chunk_emb,
                semantic_score=coherence
            ))
        return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  ENTITY EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class EntityExtractor:
    STOP_WORDS = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","can",
        "need","dare","ought","used","to","of","in","on","at","by","for","with","about",
        "against","between","into","through","during","before","after","above","below",
        "from","up","down","out","off","over","under","again","further","then","once",
        "here","there","when","where","why","how","all","both","each","few","more",
        "most","other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","just","because","as","until","while","that","this","these","those",
        "it","its","itself","they","them","their","what","which","who","whom","i","me",
        "my","myself","we","our",
    }

    def extract(self, text):
        if SPACY_AVAILABLE:
            doc = nlp(text[:5000])
            entities = list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2})
            keywords = list({chunk.root.lemma_.lower() for chunk in doc.noun_chunks
                             if chunk.root.lemma_.lower() not in self.STOP_WORDS
                             and len(chunk.root.lemma_) > 3})[:20]
        else:
            entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)))[:15]
            tokens = re.findall(r'\b[a-z]{4,}\b', text.lower())
            freq = defaultdict(int)
            for t in tokens:
                if t not in self.STOP_WORDS: freq[t] += 1
            keywords = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:20]]
        return entities, keywords


# ═══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGraphBuilder:
    def __init__(self, sim_threshold=0.55):
        self.G = nx.Graph()
        self.sim_threshold = sim_threshold

    def add_document(self, doc_id, doc_name):
        self.G.add_node(doc_id, label=doc_name[:30], node_type="document", size=25, color="#e8613c")

    def add_chunk(self, chunk):
        label = chunk.text[:50].replace("\n", " ") + "…"
        self.G.add_node(chunk.id, label=label, node_type="chunk", size=12, color="#8ba3c7",
                        doc_id=chunk.doc_id, coherence=chunk.semantic_score)
        if self.G.has_node(chunk.doc_id):
            self.G.add_edge(chunk.doc_id, chunk.id, relation="CONTAINS", weight=1.0)

    def add_entities(self, chunk):
        for ent in chunk.entities:
            eid = f"ent::{ent.lower().replace(' ', '_')}"
            if not self.G.has_node(eid):
                self.G.add_node(eid, label=ent[:25], node_type="entity", size=18, color="#4db8d4")
            self.G.add_edge(chunk.id, eid, relation="MENTIONS", weight=1.0)
        eids = [f"ent::{e.lower().replace(' ', '_')}" for e in chunk.entities]
        for i in range(len(eids)):
            for j in range(i+1, len(eids)):
                if self.G.has_edge(eids[i], eids[j]):
                    self.G[eids[i]][eids[j]]["weight"] += 0.5
                else:
                    self.G.add_edge(eids[i], eids[j], relation="CO_OCCURS", weight=0.5)

    def add_similarity_edges(self, chunks):
        ids = [c.id for c in chunks]
        embs = np.array([c.embedding for c in chunks if c.embedding is not None])
        if len(embs) < 2: return
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        normed = embs / norms; sim_matrix = normed @ normed.T
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                sim = float(sim_matrix[i, j])
                if sim >= self.sim_threshold:
                    self.G.add_edge(ids[i], ids[j], relation="SIMILAR", weight=round(sim, 3))

    def detect_communities(self):
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            return [list(c) for c in greedy_modularity_communities(self.G)]
        except Exception:
            return [[n] for n in self.G.nodes()]

    def get_subgraph(self, node_ids, hops=1):
        nodes = set(node_ids)
        for _ in range(hops):
            neighbours = set()
            for n in nodes:
                if self.G.has_node(n): neighbours.update(self.G.neighbors(n))
            nodes |= neighbours
        return self.G.subgraph(nodes).copy()

    def pagerank(self):
        try:
            return nx.pagerank(self.G, alpha=0.85, weight="weight")
        except Exception:
            return {n: 1.0 for n in self.G.nodes()}


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self, dim=384):
        self.dim = dim; self._ids = []; self._texts = []
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._matrix = None

    def _normalize(self, v): return v / (np.linalg.norm(v) + 1e-9)

    def add(self, chunk_id, text, embedding):
        vec = self._normalize(embedding).astype(np.float32)
        self._ids.append(chunk_id); self._texts.append(text)
        if FAISS_AVAILABLE:
            self._index.add(vec.reshape(1, -1))
        else:
            if self._matrix is None:
                self._matrix = vec.reshape(1, -1)
            else:
                self._matrix = np.vstack([self._matrix, vec])

    def search(self, query_embedding, top_k=10):
        if not self._ids: return []
        qvec = self._normalize(query_embedding).astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self._ids))
        if FAISS_AVAILABLE:
            scores, idxs = self._index.search(qvec, k)
            return [(self._ids[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]
        else:
            sims = (self._matrix @ qvec.T).flatten(); top = np.argsort(sims)[::-1][:k]
            return [(self._ids[i], float(sims[i])) for i in top]

    def __len__(self): return len(self._ids)


# ═══════════════════════════════════════════════════════════════════════════
#  RERANKER
# ═══════════════════════════════════════════════════════════════════════════

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = None
        if ST_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(model_name)
            except Exception:
                pass

    @staticmethod
    def _bm25_score(query, text, k1=1.5, b=0.75, avg_dl=200.0):
        query_terms = set(query.lower().split()); tokens = text.lower().split(); dl = len(tokens)
        freq = defaultdict(int)
        for t in tokens: freq[t] += 1
        score = 0.0
        for term in query_terms:
            if term in freq:
                tf = freq[term]; idf = math.log(1 + 1/(tf + 0.5))
                score += idf * (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avg_dl))
        return score

    def rerank(self, query, chunks, top_k=5):
        if not chunks: return []
        for c in chunks: c.rerank_score = self._bm25_score(query, c.text)
        if self.cross_encoder is not None:
            pairs = [(query, c.text[:512]) for c in chunks]
            try:
                ce_scores = self.cross_encoder.predict(pairs)
                for c, s in zip(chunks, ce_scores):
                    bm25_max = max((x.rerank_score for x in chunks), default=1.0) or 1.0
                    c.rerank_score = 0.6*float(s) + 0.4*(c.rerank_score/bm25_max)
            except Exception:
                pass
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class GraphRAGPipeline:
    def __init__(self, groq_api_key, groq_model="llama-3.3-70b-versatile",
                 embed_model_name="all-MiniLM-L6-v2",
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_k_retrieval=12, top_k_rerank=5, graph_hops=2):
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.top_k = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.graph_hops = graph_hops
        self.embed_model = SentenceTransformer(embed_model_name) if ST_AVAILABLE else None
        self.chunker = AdvancedSemanticChunker(embed_model=self.embed_model)
        self.extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.vector_store = VectorStore(dim=384 if ST_AVAILABLE else 128)
        self.reranker = Reranker(reranker_model)
        self.chunks = {}; self.documents = {}
        self.communities = []; self.pagerank_scores = {}

    def ingest_document(self, text, doc_name):
        doc_id = hashlib.md5(text[:500].encode()).hexdigest()[:12]
        if doc_id in self.documents: return 0, doc_id
        self.documents[doc_id] = doc_name
        self.graph_builder.add_document(doc_id, doc_name)
        raw_chunks = self.chunker.chunk(text, doc_id, doc_name)
        for chunk in raw_chunks:
            ents, kws = self.extractor.extract(chunk.text)
            chunk.entities = ents; chunk.keywords = kws
            self.chunks[chunk.id] = chunk
            self.graph_builder.add_chunk(chunk)
            self.graph_builder.add_entities(chunk)
            if chunk.embedding is not None:
                self.vector_store.add(chunk.id, chunk.text, chunk.embedding)
        doc_chunks = [c for c in raw_chunks if c.embedding is not None]
        self.graph_builder.add_similarity_edges(doc_chunks)
        self.communities = self.graph_builder.detect_communities()
        self.pagerank_scores = self.graph_builder.pagerank()
        return len(raw_chunks), doc_id

    def retrieve(self, query):
        t0 = time.time()
        q_emb = (self.embed_model.encode([query], show_progress_bar=False)[0]
                 if self.embed_model else self.chunker._embed([query])[0])
        hits = self.vector_store.search(q_emb, top_k=self.top_k)
        candidate_ids = [h[0] for h in hits]
        score_map = {h[0]: h[1] for h in hits}
        subgraph = self.graph_builder.get_subgraph(candidate_ids, hops=self.graph_hops)
        graph_chunk_ids = [n for n, d in subgraph.nodes(data=True)
                           if d.get("node_type") == "chunk" and n in self.chunks]
        expanded = list({*candidate_ids, *graph_chunk_ids})
        candidates = [self.chunks[cid] for cid in expanded if cid in self.chunks]
        for c in candidates:
            pr = self.pagerank_scores.get(c.id, 0.0)
            score_map[c.id] = score_map.get(c.id, 0.0) * (1 + pr * 5)
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
        top_entities = []
        for c in reranked[:3]: top_entities.extend(c.entities)
        top_entities = list(dict.fromkeys(top_entities))[:10]
        relevant_comms = [comm for comm in self.communities
                          if sum(1 for n in comm if n in {c.id for c in reranked}) > 0]
        graph_ctx = self._build_graph_context(reranked, top_entities, subgraph)
        latency = (time.time() - t0) * 1000
        return RetrievalResult(chunks=reranked, graph_context=graph_ctx,
                               communities=relevant_comms, retrieval_scores=score_map,
                               latency_ms=latency)

    def _build_graph_context(self, chunks, entities, subgraph):
        lines = []
        if entities:
            lines.append(f"Key entities: {', '.join(entities[:8])}")
        rels = [(u, d.get('relation',''), v) for u, v, d in subgraph.edges(data=True)
                if d.get("relation") in ("CO_OCCURS","SIMILAR")][:6]
        if rels:
            lines.append("Graph relationships: " +
                         "; ".join(f"{a} —[{r}]→ {b[:20]}" for a, r, b in rels[:4]))
        return "\n".join(lines)

    def generate(self, query, result, chat_history, system_prompt=""):
        context_parts = [f"[Chunk {i+1} | {c.doc_name} | score={c.rerank_score:.3f}]\n{c.text}"
                         for i, c in enumerate(result.chunks)]
        context = "\n\n---\n\n".join(context_parts)
        system = system_prompt or (
            "You are a precise document Q&A assistant. "
            "The context includes regular text AND extracted content from images/tables "
            "marked with [IMAGE] or [TABLE] or [TABLE-OCR] tags — treat these with equal "
            "importance as regular text, as they contain key data like metrics, results, and figures.\n"
            "Rules:\n"
            "1. If the answer is a number, percentage, or single fact — reply in ONE sentence.\n"
            "2. If the answer comes from an [IMAGE] or [TABLE] block, still cite it as [Chunk N].\n"
            "3. Only elaborate if the question genuinely requires explanation.\n"
            "4. Never repeat information already stated.\n"
            "5. If the answer is not in the chunks, say: Not found in the documents."
        )
        messages = [{"role": "system", "content": system}]
        for turn in chat_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        user_msg = f"Graph context:\n{result.graph_context}\n\nRetrieved chunks:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_msg})
        response = self.groq_client.chat.completions.create(
            model=self.groq_model, messages=messages, max_tokens=512, temperature=0.0, top_p=0.9)
        return response.choices[0].message.content

    def answer(self, query, chat_history, system_prompt=""):
        result = self.retrieve(query)
        ans = self.generate(query, result, chat_history, system_prompt)
        return ans, result


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH VISUALISER
# ═══════════════════════════════════════════════════════════════════════════

def render_graph_html(G, highlight_ids=None):
    if not PYVIS_AVAILABLE:
        return "<p style='color:#6b7f99;font-family:monospace;font-size:0.8rem;padding:20px'>Install pyvis for graph visualisation.</p>"
    net = Network(height="500px", width="100%", bgcolor="#0e1219", font_color="#e8edf5")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
    COLOR_MAP = {"document":"#e8613c","chunk":"#8ba3c7","entity":"#4db8d4","concept":"#c9a84c"}
    HIGHLIGHTED = "#e8613c"
    highlight_set = set(highlight_ids or [])
    for node, data in G.nodes(data=True):
        ntype = data.get("node_type","chunk")
        color = HIGHLIGHTED if node in highlight_set else COLOR_MAP.get(ntype,"#8ba3c7")
        size = data.get("size",12)
        if node in highlight_set: size = int(size * 1.6)
        label = data.get("label", str(node)[:20])
        net.add_node(str(node), label=label, color=color, size=size, title=f"[{ntype}] {label}")
    for u, v, data in G.edges(data=True):
        rel = data.get("relation",""); w = data.get("weight",1.0)
        edge_color = "#1c2230" if rel == "CONTAINS" else "#243040"
        net.add_edge(str(u), str(v), title=rel, width=max(0.5, w*2), color=edge_color)
    net.set_options('{"interaction":{"hover":true,"tooltipDelay":100},"physics":{"stabilization":{"iterations":150}}}')
    return net.generate_html()


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "pipeline": None,
        "chat_history": [],
        "last_result": None,
        "docs_ingested": [],
        "total_chunks": 0,
        "groq_api_key": "",                          # ← persists across sidebar collapse
        "groq_model": "llama-3.3-70b-versatile",
        "system_prompt": (
            "You are a precise document Q&A assistant. "
            "Reply in one sentence for facts/numbers. "
            "Cite chunk numbers e.g. [Chunk 2]. "
            "Only elaborate if the question needs explanation. "
            "If not found, say: Not found in the documents."
        ),
        "top_k": 5,
        "graph_hops": 2,
        "sim_threshold": 0.45,
        "show_graph": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_pipeline():
    return st.session_state.get("pipeline")


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def _build_pipeline(api_key: str, model: str):
    """Shared helper used by both sidebar and inline setup card."""
    st.session_state.pipeline = GraphRAGPipeline(
        groq_api_key=api_key,
        groq_model=model,
        top_k_retrieval=max(st.session_state.top_k * 2, 10),
        top_k_rerank=st.session_state.top_k,
        graph_hops=st.session_state.graph_hops,
    )
    st.session_state.pipeline.chunker.sim_threshold = st.session_state.sim_threshold
    st.session_state.groq_api_key = api_key
    st.session_state.groq_model = model


def sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo-mark">⬡ Graph RAG</div>
            <div class="logo-sub">Knowledge Graph · Retrieval</div>
            <div class="version-pill">v2.0 · GROQ</div>
        </div>
        """, unsafe_allow_html=True)

        # API config
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-label">API Configuration</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("<div style='padding: 0 20px 16px;'>", unsafe_allow_html=True)

            # Read back the persisted key so it survives sidebar collapse/expand
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                label_visibility="collapsed",
                help="Get yours at console.groq.com",
                value=st.session_state.groq_api_key,
                key="sidebar_groq_key",
            )
            # Always keep session state in sync
            st.session_state.groq_api_key = groq_key
            st.caption("Groq API Key  ·  console.groq.com")

            groq_models = [
                "llama-3.3-70b-versatile", "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it",
            ]
            model = st.selectbox(
                "Model", groq_models,
                index=groq_models.index(st.session_state.groq_model),
                label_visibility="collapsed",
            )
            st.caption("Inference model")
            st.session_state.groq_model = model
            st.markdown("</div>", unsafe_allow_html=True)

        # RAG params
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-label">RAG Parameters</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("<div style='padding: 0 20px 16px;'>", unsafe_allow_html=True)
            st.session_state.top_k = st.slider("Top-K chunks", 1, 10, st.session_state.top_k)
            st.session_state.graph_hops = st.slider("Graph hops", 1, 3, st.session_state.graph_hops)
            st.session_state.sim_threshold = st.slider(
                "Chunk threshold", 0.2, 0.9, st.session_state.sim_threshold, step=0.05)
            with st.expander("System Prompt"):
                st.session_state.system_prompt = st.text_area(
                    "Prompt", value=st.session_state.system_prompt,
                    height=100, label_visibility="collapsed")
            st.session_state.show_graph = st.toggle(
                "Show knowledge graph", value=st.session_state.show_graph)
            st.markdown("</div>", unsafe_allow_html=True)

        # Action buttons
        st.markdown("<div style='padding: 12px 20px; border-bottom: 1px solid var(--border);'>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        build_clicked = col1.button("⬡ Build", use_container_width=True)
        reset_clicked = col2.button("↺ Reset", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if reset_clicked:
            for k in ["pipeline", "chat_history", "last_result", "docs_ingested", "total_chunks"]:
                st.session_state[k] = ([] if k in ("chat_history", "docs_ingested")
                                       else 0 if k == "total_chunks" else None)
            st.rerun()

        if build_clicked:
            if not groq_key:
                st.error("Enter a Groq API key.")
            else:
                with st.spinner("Initialising…"):
                    try:
                        _build_pipeline(groq_key, model)
                        st.success("Pipeline ready")
                    except Exception as e:
                        st.error(f"{e}")

        # Status stats
        pipe = get_pipeline()
        if pipe:
            n_docs   = len(pipe.documents)
            n_chunks = len(pipe.chunks)
            n_nodes  = pipe.graph_builder.G.number_of_nodes()
            n_edges  = pipe.graph_builder.G.number_of_edges()
            n_comms  = len(pipe.communities)
            n_turns  = len(st.session_state.chat_history) // 2
            st.markdown(f"""
            <div class="sidebar-stats">
                <div class="sidebar-stat"><div class="v">{n_docs}</div><div class="l">Docs</div></div>
                <div class="sidebar-stat"><div class="v">{n_chunks}</div><div class="l">Chunks</div></div>
                <div class="sidebar-stat"><div class="v">{n_nodes}</div><div class="l">Nodes</div></div>
                <div class="sidebar-stat"><div class="v">{n_edges}</div><div class="l">Edges</div></div>
                <div class="sidebar-stat"><div class="v">{n_comms}</div><div class="l">Comm.</div></div>
                <div class="sidebar-stat"><div class="v">{n_turns}</div><div class="l">Turns</div></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 14px 20px; font-family: 'JetBrains Mono', monospace;
                        font-size: 0.72rem; color: var(--muted);
                        border-bottom: 1px solid var(--border);">
                Enter API key and click Build to initialise pipeline.
            </div>
            """, unsafe_allow_html=True)

        # Capabilities
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-label">Capabilities</div>
        </div>
        """, unsafe_allow_html=True)
        caps = [
            (ST_AVAILABLE,            "Sentence Embeddings"),
            (FAISS_AVAILABLE,         "FAISS Vector Index"),
            (SPACY_AVAILABLE,         "spaCy NER"),
            (PYVIS_AVAILABLE,         "Graph Visualisation"),
            (True,                    "Groq LLM Inference"),
            (PDF_BACKEND is not None, f"PDF ({PDF_BACKEND or 'unavailable'})"),
        ]
        items_html = "".join(
            f'<div class="cap-item">'
            f'<div class="cap-dot {"on" if ok else "off"}"></div>{label}</div>'
            for ok, label in caps
        )
        st.markdown(
            f'<div style="padding: 12px 20px 20px;">'
            f'<div class="cap-row">{items_html}</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE HEADER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  INLINE SETUP CARD  (shown in main area when pipeline is not built)
# ═══════════════════════════════════════════════════════════════════════════

def inline_setup_card():
    """
    Renders a compact API-key + model + Build row in the main content area.
    Visible whenever the pipeline hasn't been initialised yet — works even
    when the sidebar is fully collapsed.
    """
    st.markdown("""
    <div class="inline-setup-card">
        <div class="inline-setup-label">⚡ Quick setup — or open the sidebar</div>
    </div>
    """, unsafe_allow_html=True)

    groq_models = [
        "llama-3.3-70b-versatile", "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it",
    ]

    qcol1, qcol2, qcol3 = st.columns([3, 2, 1])

    with qcol1:
        inline_key = st.text_input(
            "Groq API Key (inline)",
            type="password",
            placeholder="gsk_...",
            value=st.session_state.groq_api_key,
            label_visibility="collapsed",
            key="inline_groq_key",
        )
        st.caption("Groq API Key · console.groq.com")

    with qcol2:
        inline_model = st.selectbox(
            "Model (inline)",
            groq_models,
            index=groq_models.index(st.session_state.groq_model),
            label_visibility="collapsed",
            key="inline_model",
        )

    with qcol3:
        inline_build = st.button(
            "⬡ Build Pipeline",
            use_container_width=True,
            key="inline_build",
        )

    if inline_build:
        key_to_use = inline_key or st.session_state.groq_api_key
        if not key_to_use:
            st.error("Enter a Groq API key.")
        else:
            with st.spinner("Initialising pipeline…"):
                try:
                    _build_pipeline(key_to_use, inline_model)
                    st.success("Pipeline ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text_from_file(uploaded_file):
    fname = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    if fname.endswith(".pdf"):
        if PDF_BACKEND is None:
            raise RuntimeError("No PDF library installed. pip install pdfplumber")
        text, meta = PDFExtractor.extract(raw_bytes, uploaded_file.name)
        meta["filename"] = uploaded_file.name
        meta["size_kb"] = round(len(raw_bytes) / 1024, 1)
        return text, meta
    elif fname.endswith((".txt", ".md")):
        text = raw_bytes.decode("utf-8", errors="replace")
        return text, {
            "filename": uploaded_file.name, "backend": "plaintext",
            "pages": text.count("\n\n") + 1,
            "size_kb": round(len(raw_bytes) / 1024, 1),
        }
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def tab_ingest():
    page_header("Document Ingestion", "Upload · Extract · Chunk · Index")
    pipe = get_pipeline()
    if not pipe:
        st.info("Initialise the pipeline first — enter your Groq API key above or in the sidebar.")
        return

    if PDF_BACKEND:
        st.markdown(f"""
        <div style="display:inline-flex; align-items:center; gap:8px;
                    background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.2);
                    border-radius:6px; padding:8px 14px;
                    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                    color:#10b981; margin-bottom:20px;">
            <span style="width:6px;height:6px;border-radius:50%;
                         background:#10b981;display:inline-block;"></span>
            PDF extraction active via {PDF_BACKEND}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No PDF library found. `pip install pdfplumber`")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        section_heading("Upload Files")
        uploaded = st.file_uploader("", type=["pdf","txt","md"],
                                    accept_multiple_files=True, label_visibility="collapsed")
        if uploaded:
            for f in uploaded:
                icon = "PDF" if f.name.lower().endswith(".pdf") else "TXT"
                sz = round(f.size / 1024, 1)
                st.markdown(f"""
                <div class="doc-card">
                    <div class="doc-icon">{icon}</div>
                    <div class="doc-info">
                        <div class="doc-name">{f.name}</div>
                        <div class="doc-meta">{sz} KB · {f.type}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("⚡  Ingest All Files", disabled=not uploaded, use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            errors = []
            for idx, f in enumerate(uploaded):
                status.info(f"Processing {f.name}…")
                try:
                    text, meta = _extract_text_from_file(f)
                    if not text.strip():
                        st.warning(f"{f.name}: no text extracted (scanned PDF?)")
                        continue
                    word_count = len(text.split())
                    status.info(f"Chunking {f.name} ({word_count:,} words)…")
                    with st.spinner(f"Building graph for {f.name}…"):
                        n_chunks, doc_id = pipe.ingest_document(text, f.name)
                    if n_chunks:
                        st.success(f"{f.name} → {n_chunks} chunks · "
                                   f"{meta.get('pages','?')} pages · {word_count:,} words")
                        st.session_state.docs_ingested.append({
                            "name": f.name, "doc_id": doc_id, "chunks": n_chunks,
                            "pages": meta.get("pages","—"), "words": word_count,
                            "backend": meta.get("backend","—"),
                            "title": meta.get("title",""), "author": meta.get("author",""),
                        })
                    else:
                        st.info(f"{f.name} already indexed.")
                except Exception as e:
                    errors.append((f.name, str(e)))
                    st.error(f"{f.name}: {e}")
                progress.progress((idx + 1) / len(uploaded))
            status.empty()
            if not errors:
                st.balloons()

    with col_right:
        section_heading("Paste Text")
        sample_text = st.text_area(
            "", height=220,
            placeholder="Paste any article, research paper excerpt, or notes…",
            label_visibility="collapsed",
        )
        doc_name = st.text_input(
            "", value="manual_input", placeholder="Document name",
            label_visibility="collapsed", key="paste_docname",
        )
        if st.button("⚡  Ingest Text", use_container_width=True) and sample_text.strip():
            with st.spinner("Chunking…"):
                n_chunks, doc_id = pipe.ingest_document(sample_text, doc_name)
            if n_chunks:
                st.success(f"Ingested {n_chunks} chunks")
                st.session_state.docs_ingested.append({
                    "name": doc_name, "doc_id": doc_id, "chunks": n_chunks,
                    "pages":"—", "words": len(sample_text.split()),
                    "backend":"plaintext", "title":"", "author":"",
                })
            else:
                st.info("Already indexed.")

        with st.expander("PDF Tips"):
            st.markdown("""
**Best results:**
- Text-based PDFs (not scanned images)
- For scanned PDFs, run OCR first

**Recommended:**
```bash
pip install pdfplumber
# or
pip install PyMuPDF
```
""")

    if st.session_state.docs_ingested:
        st.divider()
        section_heading("Ingested Documents")
        df = pd.DataFrame(st.session_state.docs_ingested)
        show_cols = [c for c in ["name","chunks","pages","words","backend","title","author"]
                     if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def tab_chat():
    page_header("Chat", "Semantic retrieval · Graph expansion · Groq inference")
    pipe = get_pipeline()
    if not pipe:
        st.info("Initialise the pipeline first.")
        return
    if not pipe.chunks:
        st.info("Ingest at least one document to begin chatting.")
        return

    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            cls = "msg-user" if turn["role"] == "user" else "msg-ai"
            st.markdown(
                f'<div class="msg-wrap"><div class="{cls}">{turn["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:var(--muted);
                    font-family:'JetBrains Mono',monospace; font-size:0.75rem; letter-spacing:1px;">
            ASK ANYTHING ABOUT YOUR DOCUMENTS
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "", key="chat_input",
            placeholder="What does the document say about…?",
            label_visibility="collapsed",
        )
    with col_btn:
        send = st.button("Send", use_container_width=True)

    if send and query.strip():
        with st.spinner("Retrieving · Reranking · Generating…"):
            try:
                answer, result = pipe.answer(
                    query, st.session_state.chat_history, st.session_state.system_prompt)
                st.session_state.chat_history.append({"role":"user","content":query})
                st.session_state.chat_history.append({"role":"assistant","content":answer})
                st.session_state.last_result = result
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.chat_history:
        if st.button("↺  Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.rerun()


def tab_retrieval():
    page_header("Retrieval Inspector", "Chunk scoring · Graph context · Entities")
    result: Optional[RetrievalResult] = st.session_state.get("last_result")
    if not result:
        st.info("Ask a question in the Chat tab to inspect retrieval details here.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks Retrieved", len(result.chunks))
    col2.metric("Latency (ms)", f"{result.latency_ms:.0f}")
    col3.metric("Communities", len(result.communities))
    col4.metric("Graph Nodes", sum(len(c) for c in result.communities))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_heading("Retrieved Chunks — After Reranking")

    for i, chunk in enumerate(result.chunks):
        score_pct = min(100, max(5, int(abs(chunk.rerank_score) * 40 + 50)))
        coh_pct   = min(100, max(5, int(chunk.semantic_score * 100)))
        with st.expander(
            f"Chunk {i+1}  ·  {chunk.doc_name}  ·  "
            f"rerank {chunk.rerank_score:.4f}  ·  coherence {chunk.semantic_score:.3f}",
            expanded=(i == 0),
        ):
            st.markdown(f"""
            <div class="score-track">
                <span class="score-track-label">Rerank</span>
                <div class="score-track-bar">
                    <div class="score-track-fill" style="width:{score_pct}%"></div>
                </div>
                <span class="score-track-val">{chunk.rerank_score:.4f}</span>
            </div>
            <div class="score-track">
                <span class="score-track-label">Coherence</span>
                <div class="score-track-bar">
                    <div class="score-track-fill" style="width:{coh_pct}%"></div>
                </div>
                <span class="score-track-val">{chunk.semantic_score:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<div style='font-size:0.85rem;line-height:1.7;color:var(--lead);"
                f"margin-top:12px;padding-top:12px;border-top:1px solid var(--border);'>"
                f"{chunk.text}</div>",
                unsafe_allow_html=True,
            )

            col_e, col_k = st.columns(2)
            with col_e:
                if chunk.entities:
                    tags = " ".join(
                        f'<span class="tag tag-entity">{e}</span>'
                        for e in chunk.entities[:6]
                    )
                    st.markdown(f"**Entities**<br>{tags}", unsafe_allow_html=True)
            with col_k:
                if chunk.keywords:
                    tags = " ".join(
                        f'<span class="tag tag-keyword">{k}</span>'
                        for k in chunk.keywords[:6]
                    )
                    st.markdown(f"**Keywords**<br>{tags}", unsafe_allow_html=True)

    if result.graph_context:
        st.divider()
        section_heading("Graph Context Passed to LLM")
        st.markdown(
            f'<div class="graph-ctx">{result.graph_context}</div>',
            unsafe_allow_html=True,
        )


def tab_graph():
    page_header("Knowledge Graph", "Nodes · Edges · Communities · PageRank")
    pipe = get_pipeline()
    if not pipe or not pipe.graph_builder.G.nodes():
        st.info("Ingest documents to build the knowledge graph.")
        return

    G = pipe.graph_builder.G
    highlight_ids = (
        [c.id for c in st.session_state.last_result.chunks]
        if st.session_state.last_result else []
    )

    col1, col2, _ = st.columns([2, 3, 1])
    with col1:
        max_nodes = st.slider("Max nodes", 20, 300, 80)
    with col2:
        node_types = st.multiselect(
            "Node types", ["document","chunk","entity"],
            default=["document","chunk","entity"],
        )

    nodes_to_show = [n for n, d in G.nodes(data=True)
                     if d.get("node_type","chunk") in node_types]
    nodes_to_show = sorted(
        nodes_to_show,
        key=lambda n: (0 if n in highlight_ids else 1, -pipe.pagerank_scores.get(n, 0)),
    )[:max_nodes]
    sub = G.subgraph(nodes_to_show).copy()
    html_str = render_graph_html(sub, highlight_ids)
    st.components.v1.html(html_str, height=520, scrolling=False)

    st.markdown("""
    <div style="display:flex; gap:20px; margin-top:10px;
                font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--muted);">
        <span style="color:#e8613c">⬡ Document</span>
        <span style="color:#8ba3c7">⬡ Chunk</span>
        <span style="color:#4db8d4">⬡ Entity</span>
        <span style="color:#e8613c; opacity:0.6">⬡ Retrieved</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    section_heading("Community Detection")
    comms = pipe.communities[:10]
    rows = []
    for i, comm in enumerate(comms):
        types = [G.nodes[n].get("node_type","?") for n in comm if G.has_node(n)]
        rows.append({
            "Community": f"#{i+1}",
            "Size": len(comm),
            "Types": ", ".join(sorted(set(types))),
            "Top PR node": max(
                comm, key=lambda n: pipe.pagerank_scores.get(n, 0), default="—"
            )[:35],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_analytics():
    page_header("Analytics", "Distributions · PageRank · Chunk explorer")
    pipe = get_pipeline()
    if not pipe or not pipe.chunks:
        st.info("Ingest documents to see analytics.")
        return

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly for charts: `pip install plotly`")
        return

    G = pipe.graph_builder.G
    chunks = list(pipe.chunks.values())
    PLOT_THEME = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#8ba3c7", title_font_size=13,
        title_font_family="JetBrains Mono", font_family="JetBrains Mono",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            x=[c.semantic_score for c in chunks], nbins=20,
            title="CHUNK COHERENCE DISTRIBUTION",
            labels={"x":"Coherence","y":"Count"},
            color_discrete_sequence=["#8ba3c7"],
        )
        fig.update_layout(**PLOT_THEME, showlegend=False)
        fig.update_traces(marker_line_color="#1c2230", marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        doc_chunk_counts = defaultdict(int)
        for c in chunks: doc_chunk_counts[c.doc_name] += 1
        fig2 = px.bar(
            x=list(doc_chunk_counts.keys()), y=list(doc_chunk_counts.values()),
            title="CHUNKS PER DOCUMENT",
            labels={"x":"Document","y":"Chunks"},
            color_discrete_sequence=["#e8613c"],
        )
        fig2.update_layout(**PLOT_THEME, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        type_counts = defaultdict(int)
        for _, d in G.nodes(data=True): type_counts[d.get("node_type","unknown")] += 1
        fig3 = px.pie(
            names=list(type_counts.keys()), values=list(type_counts.values()),
            title="GRAPH NODE TYPES",
            color_discrete_sequence=["#e8613c","#8ba3c7","#4db8d4","#c9a84c"],
            hole=0.5,
        )
        fig3.update_layout(**PLOT_THEME)
        fig3.update_traces(textfont_color="#e8edf5")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        pr = pipe.pagerank_scores
        top_pr = sorted(pr.items(), key=lambda x: -x[1])[:15]
        labels = [G.nodes[n].get("label", n)[:22] if G.has_node(n) else n[:22] for n, _ in top_pr]
        values = [v for _, v in top_pr]
        fig4 = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#4db8d4"))
        fig4.update_layout(**PLOT_THEME, title="TOP NODES BY PAGERANK",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    section_heading("All Chunks")
    rows = [{
        "ID": c.id[:8], "Document": c.doc_name, "Index": c.chunk_index,
        "Tokens": len(c.text.split()), "Coherence": round(c.semantic_score, 3),
        "Entities": len(c.entities), "Keywords": ", ".join(c.keywords[:4]),
        "Preview": c.text[:80].replace("\n"," ") + "…",
    } for c in chunks]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_about():
    page_header("About", "Architecture · Stack · Setup")
    st.markdown("""
<div style="max-width: 760px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);
            letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;">
    Graph RAG extends standard RAG by building and querying a knowledge graph
    alongside a vector store, enabling relationship-aware retrieval.
</div>
</div>
""", unsafe_allow_html=True)

    section_heading("Architecture")
    st.code("""
Documents
    │
    ▼
Advanced Semantic Chunker
  · Sentence splitting (spaCy)
  · Embedding-based boundary detection
  · Overlap + coherence scoring
    │
    ├──► FAISS Vector Store  ◄── Query Embedding
    │                               │
    ├──► Knowledge Graph      ──────┘  ANN search + Graph expansion
    │    · Document nodes
    │    · Chunk nodes        PageRank boosting
    │    · Entity nodes            │
    │    · CO_OCCURS / SIMILAR     ▼
    │    · Community detection  Candidate Chunks
    │                               │
    └──────────────────  Cross-Encoder Reranker (sentence-transformers)
                                    │
                           Top-K Chunks + Graph Context
                                    │
                             Groq LLM (llama-3.3-70b)
                                    │
                              Final Answer
""", language="text")

    section_heading("Stack")
    data = {
        "Component": ["Embeddings","Cross-encoder Reranking","Vector Index",
                       "Knowledge Graph","Graph Visualisation","NER / Sentence Split","LLM","UI"],
        "Library":   ["sentence-transformers","sentence-transformers","faiss-cpu",
                      "networkx","pyvis","spacy","groq (Groq API)","streamlit"],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    section_heading("Setup")
    st.code("""pip install streamlit groq networkx sentence-transformers faiss-cpu \\
            spacy numpy pandas plotly pyvis python-dotenv
python -m spacy download en_core_web_sm
streamlit run graph_rag_app.py""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    sidebar()

    # Top-level app header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:4px;">
        <div style="font-style:italic; font-weight:300;
                    font-size:2.2rem; color:var(--lead); letter-spacing:-1px; line-height:1;">
            Graph RAG
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--muted);
                    letter-spacing:2px; text-transform:uppercase; padding-top:6px;">
            Semantic Chunking · Knowledge Graphs · Cross-Encoder Reranking · Groq LLM
        </div>
    </div>
    <div style="height:1px; background:var(--border); margin-bottom:20px;"></div>
    """, unsafe_allow_html=True)

    # ── Inline setup card — visible whenever pipeline is not built ──────────
    # Works even when the sidebar is collapsed.
    if not get_pipeline():
        inline_setup_card()

    # ── Main tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "  Ingest  ", "  Chat  ", "  Retrieval  ",
        "  Graph  ", "  Analytics  ", "  About  ",
    ])
    with tabs[0]: tab_ingest()
    with tabs[1]: tab_chat()
    with tabs[2]: tab_retrieval()
    with tabs[3]: tab_graph()
    with tabs[4]: tab_analytics()
    with tabs[5]: tab_about()


if __name__ == "__main__":
    main()