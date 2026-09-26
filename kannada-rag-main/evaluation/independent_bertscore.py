"""
evaluation/independent_bertscore.py

Computes a SECOND BERTScore using a model outside the KN-BERT family,
to check whether the current evaluator (l3cube-pune/kannada-sentence-bert-nli
— the same model used for KN-BERT retrieval) is inflating KN-BERT's
apparent advantage through evaluator-retriever coupling.

Recommended independent model: a general multilingual BERT not tied to
any retrieval configuration in this study, e.g. bert-base-multilingual-cased
or google/muril-base-cased. Avoid intfloat/multilingual-e5-* — that IS
one of your retrieval configurations (E5), so using it as the evaluator
would just move the coupling problem rather than fix it.

USAGE:
  python evaluation/independent_bertscore.py \
      --results results_pipelineB_HONEST_final_v2.json \
      --model bert-base-multilingual-cased \
      --out results_with_independent_bertscore.json
"""
import os, sys, json, argparse

def run(results_path, model_name, out_path):
    from bert_score import score as bertscore  # pip install bert-score

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    all_sections = {**data.get("combos", {}), **data.get("no_rag", {})}

    print(f"Loading independent evaluator model: {model_name}")
    print("(This is a different model family from KN-BERT retrieval — that's the point.)\n")

    for label, rows in all_sections.items():
        answered_rows = [r for r in rows if not r.get("skipped") and r.get("answer")]
        if not answered_rows:
            continue
        preds = [r["answer"] for r in answered_rows]
        refs  = [r["reference"] for r in answered_rows]

        print(f"Scoring {label} ({len(preds)} answers)...")
        P, R, F1 = bertscore(preds, refs, model_type=model_name, lang="kn", verbose=False)

        for row, f1_val in zip(answered_rows, F1.tolist()):
            row["bert_f1_independent"] = round(f1_val, 4)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nWrote -> {out_path}")

    # Quick comparison summary
    print(f"\n{'='*70}\nORIGINAL vs INDEPENDENT EVALUATOR — mean BERT-F1 per config\n{'='*70}")
    for label, rows in all_sections.items():
        answered_rows = [r for r in rows if not r.get("skipped") and "bert_f1_independent" in r]
        if not answered_rows:
            continue
        orig = sum(r.get("bert_f1", 0) for r in answered_rows) / len(answered_rows)
        indep = sum(r.get("bert_f1_independent", 0) for r in answered_rows) / len(answered_rows)
        print(f"  {label:<50} original={orig:.4f}  independent={indep:.4f}  diff={indep-orig:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--model", default="bert-base-multilingual-cased")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.results, args.model, args.out)
