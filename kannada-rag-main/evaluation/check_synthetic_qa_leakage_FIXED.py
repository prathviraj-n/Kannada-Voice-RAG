"""
evaluation/check_synthetic_qa_leakage.py

Checks whether any benchmark question (or a close paraphrase) appears
among the synthetic QA pairs generated during metadata-enriched chunking
and stored in the index. If a benchmark question is too similar to a
synthetic QA pair already sitting in the retrieval index, that question
is not a fair test of retrieval — the system could trivially match on
memorized surface form rather than genuine semantic retrieval.

Two similarity checks are run (no GPU/embedding model required):
  1. Exact/near-exact string match (normalized).
  2. Token-overlap (Jaccard) similarity, flagging anything above a
     configurable threshold for manual review.

USAGE:
  python evaluation/check_synthetic_qa_leakage.py \
      --gt evaluation/kannada_ground_truth_76.json \
      --index_dir index/faiss \
      --qa_column synthetic_qa \
      --threshold 0.6
"""
import os, sys, json, argparse, unicodedata, re
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[^\u0C80-\u0CFF\s]", " ", text)
    return " ".join(text.split()).strip()


def tokenize(text):
    return set(normalize(text).split())


def jaccard(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def find_qa_column(df):
    """Try common naming conventions for the synthetic-QA metadata field."""
    candidates = ["synthetic_questions", "synthetic_qa", "qa_pairs", "generated_qa", "synth_qa", "syntheticQA", "qa"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_questions_from_qa_field(value):
    """
    The synthetic QA field's exact structure is unknown ahead of time —
    handle common shapes: a dict {"question":..., "answer":...}, a list
    or numpy array of such dicts/strings, or a raw string containing the
    question text. Confirmed real schema (2026-09): a numpy array of
    plain question strings under column name 'synthetic_questions'.
    """
    import numpy as np
    questions = []
    if value is None:
        return questions
    if isinstance(value, dict):
        q = value.get("question") or value.get("Q") or value.get("q")
        if q:
            questions.append(str(q))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            questions.extend(extract_questions_from_qa_field(item))
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return extract_questions_from_qa_field(parsed)
        except Exception:
            questions.append(value)
    return questions


def run(gt_path, index_dir, qa_column, threshold):
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    pdf_bases = sorted(set(q.get("source_pdf", "Kan-Science") for q in gt))
    all_synthetic_questions = []  # list of (pdf_base, chunk_id, synthetic_question_text)

    for pdf_base in pdf_bases:
        path = os.path.join(index_dir, pdf_base, "chunks.parquet")
        if not os.path.exists(path):
            print(f"  ⚠️  No chunks.parquet for '{pdf_base}' at {path} — skipping")
            continue
        df = pd.read_parquet(path)

        col = qa_column if qa_column in df.columns else find_qa_column(df)
        if col is None:
            print(f"  ⚠️  No synthetic-QA column found in '{pdf_base}' chunks.parquet.")
            print(f"      Available columns: {df.columns.tolist()}")
            print(f"      If synthetic QA generation uses a different field name, pass --qa_column explicitly.")
            continue

        print(f"  Found synthetic-QA column '{col}' in '{pdf_base}' ({len(df)} chunks)")
        for idx, row in df.iterrows():
            questions = extract_questions_from_qa_field(row.get(col))
            for q in questions:
                all_synthetic_questions.append((pdf_base, idx, q))

    print(f"\nTotal synthetic QA questions found across all indexes: {len(all_synthetic_questions)}")
    if not all_synthetic_questions:
        print("\n⚠️  Could not locate any synthetic QA content — cannot complete leakage check.")
        print("    This likely means synthetic QA generation is either not yet implemented in your")
        print("    ingestion pipeline, or stored under a field name this script doesn't recognize.")
        print("    Report this honestly in the manuscript rather than claim a leakage check was performed.")
        return

    # Precompute token sets for synthetic questions
    synth_tokenized = [(pdf, cid, q, tokenize(q)) for pdf, cid, q in all_synthetic_questions]

    flagged = []
    exact_matches = []

    for item in gt:
        qid = item["id"]
        question = item["question"]
        q_norm = normalize(question)
        q_tokens = tokenize(question)

        for pdf, cid, synth_q, synth_tokens in synth_tokenized:
            if normalize(synth_q) == q_norm:
                exact_matches.append((qid, question, pdf, cid, synth_q, 1.0))
                continue
            sim = jaccard(q_tokens, synth_tokens)
            if sim >= threshold:
                flagged.append((qid, question, pdf, cid, synth_q, sim))

    print(f"\n{'='*70}\nLEAKAGE CHECK RESULTS\n{'='*70}")
    print(f"Exact matches: {len(exact_matches)}")
    for qid, q, pdf, cid, synth_q, sim in exact_matches:
        print(f"  [{qid}] EXACT MATCH in {pdf} chunk {cid}")
        print(f"    benchmark: {q}")
        print(f"    synthetic: {synth_q}")

    print(f"\nNear-duplicate matches (Jaccard >= {threshold}): {len(flagged)}")
    for qid, q, pdf, cid, synth_q, sim in sorted(flagged, key=lambda x: -x[5]):
        print(f"  [{qid}] similarity={sim:.2f} in {pdf} chunk {cid}")
        print(f"    benchmark: {q}")
        print(f"    synthetic: {synth_q}")

    if not exact_matches and not flagged:
        print("\n✅ No leakage detected at this threshold. Safe to report a clean leakage-check result.")
    else:
        print(f"\n⚠️  {len(exact_matches) + len(flagged)} question(s) flagged for manual review.")
        print("    For each, either remove/regenerate the synthetic QA pair before re-indexing,")
        print("    or exclude the affected benchmark question from retrieval-grounding claims.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--index_dir", default="index/faiss")
    p.add_argument("--qa_column", default=None, help="Explicit column name if auto-detection fails")
    p.add_argument("--threshold", type=float, default=0.6, help="Jaccard similarity threshold to flag")
    args = p.parse_args()
    run(args.gt, args.index_dir, args.qa_column, args.threshold)
