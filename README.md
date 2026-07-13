# GraphRAG Intelligence

A Graph-based Retrieval-Augmented Generation (RAG) system that combines semantic chunking, knowledge-graph construction, and hybrid reranking to answer questions over your own documents — powered by Groq's LLaMA models and deployed with Streamlit.

**Live demo:** _add your Streamlit Cloud URL here once deployed_

---

## Features

- **Advanced semantic chunking** — splits documents into coherent chunks based on embedding similarity rather than fixed character windows
- **Knowledge graph construction** — extracts entities (spaCy) and builds a NetworkX graph of relationships across your documents, with community detection (greedy modularity) to surface topic clusters
- **Hybrid retrieval** — dense vector search (FAISS + Sentence-Transformers) combined with BM25 keyword scoring, then reranked with a cross-encoder for final relevance ordering
- **Scanned PDF support** — automatic OCR fallback for image-based tables and regions using Tesseract, with adaptive image preprocessing (contrast/sharpening/threshold retries) so low-quality scans still extract cleanly
- **Multi-backend PDF extraction** — tries `pdfplumber` → `pymupdf` → `pypdf` → `PyPDF2` in order, so the app degrades gracefully depending on what's installed
- **Interactive graph visualization** — explore the extracted knowledge graph via PyVis
- **Groq-powered generation** — fast LLM responses using Groq's LLaMA models
- **"Glass Intelligence" UI** — custom dark glassmorphism theme built in Streamlit

## Tech Stack

| Layer | Tools |
|---|---|
| LLM | Groq API (LLaMA) |
| Embeddings & Reranking | Sentence-Transformers, CrossEncoder |
| Vector Search | FAISS |
| Keyword Search | BM25 (custom implementation) |
| Knowledge Graph | NetworkX, spaCy (NER), greedy modularity community detection |
| PDF/OCR | pdfplumber, PyMuPDF, pypdf, PyPDF2, Tesseract (pytesseract) |
| Visualization | PyVis, Plotly |
| Frontend | Streamlit |

## Project Structure

```
.
├── app.py            # Main Streamlit application
├── requirements.txt   # Python dependencies
├── packages.txt        # System-level apt dependencies (Tesseract OCR)
└── README.md
```

## Running Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if not already pulled via requirements.txt
streamlit run app.py
```

You'll also need [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally if you want scanned-PDF support:
- **Windows:** install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract.exe` is on your PATH
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

Enter your Groq API key in the sidebar (or set it as an environment variable) and click **Build** to initialize the pipeline.

## Deployment (Streamlit Community Cloud)

1. Push `app.py`, `requirements.txt`, and `packages.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to the repo, branch, and `app.py`
3. (Optional) Add `GROQ_API_KEY` under **Advanced settings → Secrets** so it doesn't need to be re-entered each session
4. Deploy

## How It Works

1. **Ingest** — upload PDF/TXT/MD files; text is extracted (with OCR fallback for scanned tables/images) and split into semantically coherent chunks
2. **Index** — chunks are embedded and stored in a FAISS vector index; entities are extracted and added as nodes/edges to a knowledge graph
3. **Query** — a question is embedded and matched against both the vector index (semantic) and BM25 (keyword), merged, and reranked with a cross-encoder
4. **Generate** — the top reranked chunks are passed to Groq's LLaMA model as context to produce a grounded answer

## License

_Add a license (e.g. MIT) if you plan to make this public._
