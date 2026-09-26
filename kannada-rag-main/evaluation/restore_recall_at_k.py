"""
evaluation/restore_recall_at_k.py

recompute_metrics.py tries to recompute recall_at_k from a row's
"retrieved_texts" field — but compare_eval_v2.py never actually saved
that field to disk (only the already-computed recall_at_k NUMBER was
saved). So recompute_metrics.py silently falls back to 0.0 for every
row, overwriting the original, valid recall_at_k values.

This script restores the original recall_at_k values from the
pre-recompute file into the post-recompute (ROUGE-fixed) file, so you
get BOTH the corrected ROUGE/METEOR/BLEU AND the original valid
recall_at_k in one final file.

USAGE:
  python evaluation/restore_recall_at_k.py \
      --pre_recompute results_pipelineB_TRULY_final.json \
      --post_recompute results_pipelineB_TRULY_final_fixed.json \
      --out results_pipelineB_ACTUALLY_final.json
"""
import json, argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pre_recompute", required=True)
    p.add_argument("--post_recompute", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.pre_recompute, encoding="utf-8") as f:
        pre = json.load(f)
    with open(args.post_recompute, encoding="utf-8") as f:
        post = json.load(f)

    restored_count = 0
    zero_still_count = 0

    for section in ("combos", "no_rag"):
        pre_section = pre.get(section, {})
        post_section = post.get(section, {})
        for label, post_rows in post_section.items():
            pre_rows = {r["id"]: r for r in pre_section.get(label, [])}
            for row in post_rows:
                qid = row.get("id")
                pre_row = pre_rows.get(qid)
                if pre_row and pre_row.get("recall_at_k", 0) > 0 and row.get("recall_at_k", 0) == 0:
                    row["recall_at_k"] = pre_row["recall_at_k"]
                    restored_count += 1
                elif row.get("recall_at_k", 0) == 0:
                    zero_still_count += 1  # genuinely 0 (e.g. No-RAG has no retrieval) or pre-value also missing

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(post, f, ensure_ascii=False, indent=2)

    print(f"Restored recall_at_k for {restored_count} rows.")
    print(f"{zero_still_count} rows still show 0.0 (expected for No-RAG rows — they have no retrieval at all).")
    print(f"Wrote -> {args.out}")
