# Kannada RAG — Voice-Enabled Educational Question Answering

A modular Retrieval-Augmented Generation (RAG) system for answering questions over
Kannada-medium educational textbooks, with speech-output synthesis for accessibility.

The system ingests scanned or digitally-native PDF textbooks, builds hybrid dense +
sparse retrieval indexes using Kannada-specific and multilingual embeddings, generates
grounded answers using Groq-hosted LLMs, and reads the answer aloud via text-to-speech.

For evaluation methodology, benchmark composition, and full results, see
[`EVALUATION.md`](EVALUATION.md).

---

## Features

- **Hybrid PDF ingestion** — native text extraction (PyMuPDF) with automatic Tesseract
  OCR fallback for scanned or non-Unicode source documents
- **Metadata-enriched chunking** — each chunk is tagged with intent type, domain,
  difficulty, language-quality score, keywords, and template-derived question variants
- **Dual-embedding retrieval** — Kannada-specific KN-BERT and multilingual E5
  embeddings, each with hybrid FAISS + BM25 search
- **Multi-agent orchestration** — query understanding, metadata-aware reranking,
  answer generation, and an evaluation-triggered retry loop
- **Groq-hosted generation** — `openai/gpt-oss-120b` / `openai/gpt-oss-20b`
- **Speech output** — Azure Edge-TTS reads generated answers aloud
- **React frontend** — chat interface with a retrieval-configuration comparison panel

---

## Project structure

```
.
├── backend/                   FastAPI application
│   ├── main.py                 API routes: /upload_pdf, /query, /tts, ...
│   ├── rag_pipeline.py          Wraps the core orchestrator
│   ├── compare_endpoint.py      Powers the frontend's retrieval-comparison panel
│   └── tts_service.py           Azure Edge-TTS wrapper
│
├── src/
│   ├── ingest_pdf.py            PDF → chunks (native text → OCR fallback)
│   ├── ocr_utils.py             Tesseract OCR + image preprocessing
│   ├── utils_text.py            Cleaning, sentence-aware chunking
│   ├── build_faiss_index.py     Builds KN-BERT and E5 FAISS + BM25 indexes
│   ├── bm25_store.py
│   ├── generate_answer_groq.py  Groq generation call (used by the live app)
│   │
│   ├── core/orchestrator.py     Full multi-agent pipeline
│   ├── agents/                  query_agent, retrieval_agent, reranker_agent,
│   │                             answer_agent, evaluation_agent, retry_agent
│   └── compare/compare_orchestrator.py   Retrieval-and-generation core used for
│                                          controlled evaluation comparisons
│
├── evaluation/                 Benchmark, evaluation harness, and analysis scripts
│   └── (see EVALUATION.md for details)
│
├── scripts/
│   └── ingest_and_index.py     Standalone ingestion + indexing entry point
│
├── frontend/                   React + Vite chat UI
├── download_models.py          Standalone embedding-model downloader
├── convert_to_safetensors.py   Checkpoint format converter
├── requirements.txt
└── EVALUATION.md               Evaluation methodology and results
```

---

## Setup

### 1. System dependencies

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-kan poppler-utils
```

If you don't have root access (e.g. on a shared HPC cluster), install via conda
instead:
```bash
conda install -c conda-forge poppler -y
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"
```

### 3. Indic NLP sentence-tokenizer resources

The `indic-nlp-library` package needs a separate resources repository (not installed
via pip):
```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git src/indic_resources
```

### 4. Embedding models

```bash
python3 download_models.py
```
This fetches `l3cube-pune/kannada-sentence-bert-nli` and
`intfloat/multilingual-e5-small` into `models/` and `backend/models/` respectively.

### 5. Environment variables

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

---

## Usage

### Ingest a PDF

```bash
mkdir -p data/raw
cp your_textbook.pdf data/raw/
python3 scripts/ingest_and_index.py your_textbook.pdf
```

This extracts text (with OCR fallback), builds both KN-BERT and E5 retrieval indexes,
and writes them to `index/faiss/<pdf_name>/`.

### Run the backend

```bash
uvicorn backend.main:app --reload
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### Query via the API directly

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "ಆಹಾರದಲ್ಲಿ ವೈವಿಧ್ಯ ಎಂದರೆ ಏನು?", "pdf_name": "Kan-Science"}'
```

---

## Evaluation

```bash
python3 evaluation/compare_eval_v2.py \
    --gt evaluation/kannada_ground_truth_76.json \
    --all_combos --mode all \
    --out results.json
```

See [`EVALUATION.md`](EVALUATION.md) for the full methodology, benchmark composition,
significance-testing framework, and results, including:

- Coverage-honest metric aggregation across all retrieval/generation configurations
- Paired bootstrap and Wilcoxon significance testing with Holm-Bonferroni correction
- An independent-evaluator robustness check
- A human faithfulness assessment with inter-rater agreement reporting
- A synthetic-question / benchmark leakage check

---

## Citation

If you use this system or benchmark in your work, please cite:

```bibtex
@article{kannada-rag-2026,
  title   = {A Voice-Enabled Modular Retrieval-Augmented Generation Framework
             for Kannada Educational Question Answering Using Hybrid Retrieval},
  author  = {Nagaraja, Prathviraj and Shetty, Sharanya and Jesika, Risha and Raghudathesh, G P},
  year    = {2026}
}
```

## License

See [LICENSE](LICENSE) for details.
