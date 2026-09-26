"""
evaluation/merge_partial_runs.py

Merges multiple compare_eval_v2.py output JSONs into one, for when a run
had to be split across days due to Groq's per-model daily token limit.
Later files' combo labels overwrite earlier ones if there's a name clash
(so pass files in chronological order — most recent/most complete wins).

USAGE:
  python evaluation/merge_partial_runs.py \
      --inputs results_pipelineB_76q.json results_pipelineB_76q_day2.json \
      --out results_pipelineB_76q_final.json
"""
import json, argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    combined_combos = {}
    combined_no_rag = {}

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for label, rows in d.get("combos", {}).items():
            answered = sum(1 for r in rows if not r.get("skipped"))
            existing = combined_combos.get(label)
            existing_answered = sum(1 for r in existing if not r.get("skipped")) if existing else -1
            if answered > existing_answered:
                combined_combos[label] = rows
                print(f"  {label}: using version from {path} ({answered}/{len(rows)} answered)")
        for label, rows in d.get("no_rag", {}).items():
            answered = sum(1 for r in rows if not r.get("skipped"))
            existing = combined_no_rag.get(label)
            existing_answered = sum(1 for r in existing if not r.get("skipped")) if existing else -1
            if answered > existing_answered:
                combined_no_rag[label] = rows
                print(f"  {label}: using version from {path} ({answered}/{len(rows)} answered)")

    output = {"meta": {"merged_from": args.inputs}, "combos": combined_combos, "no_rag": combined_no_rag}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote merged result -> {args.out}")
