"""
evaluation/build_human_rating_sheet.py

Generates a stratified sample (by difficulty and by RAG vs No-RAG) for
human faithfulness/correctness rating, and exports a CSV two annotators
can independently fill in — no coding needed from them.

Sampling: draws N questions per difficulty tier, each shown once for the
best RAG config and once for the best No-RAG config (paired), so raters
can compare grounded vs ungrounded answers to the same question blind
to which is which (columns are randomly shuffled per row).

USAGE:
  python evaluation/build_human_rating_sheet.py \
      --results results_pipelineB_HONEST_final_v2.json \
      --rag_label "kannada-bert | hybrid | openai/gpt-oss-120b" \
      --norag_label "NoRAG|openai/gpt-oss-120b" \
      --n_per_tier 10 \
      --out human_rating_sheet.csv
"""
import json, csv, random, argparse

RUBRIC_COLUMNS = [
    "context_relevance_1to5",       # only meaningful for the RAG answer; leave blank for No-RAG
    "faithfulness",                  # one of: supported / partially_supported / contradicted / unsupported
    "answer_correctness_1to5",
    "kannada_fluency_1to5",
    "appropriate_refusal_yn",        # did it correctly say "not found" if genuinely unanswerable? y/n/na
    "annotator_notes",
]

def run(results_path, rag_label, norag_label, n_per_tier, out_path, seed=42):
    random.seed(seed)
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    all_sections = {**data.get("combos", {}), **data.get("no_rag", {})}
    rag_rows = {r["id"]: r for r in all_sections[rag_label] if not r.get("skipped")}
    norag_rows = {r["id"]: r for r in all_sections[norag_label] if not r.get("skipped")}
    common_ids = set(rag_rows) & set(norag_rows)

    by_tier = {"easy": [], "medium": [], "hard": []}
    for qid in common_ids:
        diff = rag_rows[qid].get("difficulty", "medium")
        if diff in by_tier:
            by_tier[diff].append(qid)

    sampled_ids = []
    for tier, ids in by_tier.items():
        random.shuffle(ids)
        sampled_ids.extend(ids[:n_per_tier])

    print(f"Sampled {len(sampled_ids)} questions "
          f"({', '.join(f'{t}={min(n_per_tier, len(ids))}' for t, ids in by_tier.items())})")

    sheet_rows = []
    for qid in sampled_ids:
        rag_row, norag_row = rag_rows[qid], norag_rows[qid]
        question = rag_row["question"]
        reference = rag_row["reference"]

        # Two blinded rows per question — order randomized per question so
        # annotators can't tell RAG from No-RAG just by row position.
        entries = [
            {"system_id": "A", "answer": rag_row["answer"], "_true_system": "RAG"},
            {"system_id": "B", "answer": norag_row["answer"], "_true_system": "NoRAG"},
        ]
        random.shuffle(entries)

        for entry in entries:
            row = {
                "question_id": qid,
                "difficulty": rag_row.get("difficulty", ""),
                "question": question,
                "reference_answer": reference,
                "system_id": entry["system_id"],        # A/B shown to annotator (blinded)
                "system_answer": entry["answer"],
                "_true_system_DO_NOT_SHOW_ANNOTATOR": entry["_true_system"],  # for your own scoring only
            }
            for col in RUBRIC_COLUMNS:
                row[col] = ""
            sheet_rows.append(row)

    fieldnames = ["question_id", "difficulty", "question", "reference_answer",
                  "system_id", "system_answer"] + RUBRIC_COLUMNS + ["_true_system_DO_NOT_SHOW_ANNOTATOR"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sheet_rows)

    print(f"\nWrote {len(sheet_rows)} rows -> {out_path}")
    print("\nBefore sending to annotators:")
    print("  1. Delete or hide the '_true_system_DO_NOT_SHOW_ANNOTATOR' column — keep your own copy with it intact.")
    print("  2. Give annotators the rubric definitions (faithfulness categories, 1-5 scales) separately.")
    print("  3. Have each annotator fill their own COPY of this file independently (no shared editing).")
    print("  4. Once both are back, use evaluation/compute_rating_agreement.py to merge and score them.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--rag_label", required=True)
    p.add_argument("--norag_label", required=True)
    p.add_argument("--n_per_tier", type=int, default=10)
    p.add_argument("--out", default="human_rating_sheet.csv")
    args = p.parse_args()
    run(args.results, args.rag_label, args.norag_label, args.n_per_tier, args.out)
