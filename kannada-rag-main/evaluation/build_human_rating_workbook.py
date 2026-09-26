"""
evaluation/build_human_rating_workbook.py

Same sampling logic as build_human_rating_sheet.py, but outputs a
proper .xlsx workbook instead of a raw CSV:
  - Tab 1 "Instructions": the full rubric, so raters never need a
    separate document.
  - Tab 2 "Ratings": the sampled questions/answers, with dropdown data
    validation on every categorical field (faithfulness,
    appropriate_refusal_yn) and restricted integer ranges (1-5) on the
    numeric scales — raters cannot type an invalid value.
  - The hidden ground-truth column (_true_system) is on a separate,
    hidden sheet, not visible to raters at all.

USAGE:
  python evaluation/build_human_rating_workbook.py \
      --results results_pipelineB_HONEST_final_v2.json \
      --rag_label "kannada-bert | hybrid | openai/gpt-oss-120b" \
      --norag_label "NoRAG|openai/gpt-oss-120b" \
      --n_per_tier 10 \
      --out human_rating_workbook.xlsx

Then make TWO COPIES of the resulting file (one per rater) before
sending — do not let both raters edit the same file, or you lose the
independence needed to measure inter-rater agreement.
"""
import json, random, argparse
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

FAITHFULNESS_OPTIONS = ["supported", "partially_supported", "contradicted", "unsupported"]
REFUSAL_OPTIONS = ["y", "n", "na"]

RUBRIC_TEXT = [
    ("Kannada RAG — Human Faithfulness Rating: Instructions", True, 16),
    ("", False, 11),
    ("You are rating answers to Kannada educational questions. For each question you will see "
     "TWO system answers, labeled A and B — you are NOT told which system produced which answer. "
     "Rate each one independently using the columns on the 'Ratings' tab.", False, 12),
    ("", False, 11),
    ("Column: faithfulness", True, 13),
    ("Choose exactly one, from the dropdown in that cell:", False, 12),
    ("  • supported — every claim in the answer is backed by the reference answer / retrieved context.", False, 11),
    ("  • partially_supported — some claims are backed, but the answer adds unsupported details or is incomplete.", False, 11),
    ("  • contradicted — the answer states something that conflicts with the reference/context.", False, 11),
    ("  • unsupported — the answer is unrelated to, or cannot be checked against, the reference/context.", False, 11),
    ("", False, 11),
    ("Columns: answer_correctness_1to5, kannada_fluency_1to5, context_relevance_1to5", True, 13),
    ("Rate each on a 1–5 scale (dropdown provided):", False, 12),
    ("  1 = poor   2 = below average   3 = adequate   4 = good   5 = excellent", False, 11),
    ("  answer_correctness: does the answer correctly address what was asked?", False, 11),
    ("  kannada_fluency: is the Kannada natural and grammatically sound (independent of correctness)?", False, 11),
    ("  context_relevance: (leave blank if you cannot tell) how relevant is the answer to the actual question?", False, 11),
    ("", False, 11),
    ("Column: appropriate_refusal_yn", True, 13),
    ("Only fill this in if the reference answer itself indicates the question is unanswerable from the source "
     "text. Otherwise, select 'na'.", False, 12),
    ("  y — the system correctly said it could not find relevant information.", False, 11),
    ("  n — the system should have refused but instead guessed/fabricated an answer.", False, 11),
    ("  na — not applicable; the question is genuinely answerable.", False, 11),
    ("", False, 11),
    ("Column: annotator_notes", True, 13),
    ("Free text — anything you want to flag (e.g. OCR garbage in the context, ambiguous question wording).", False, 12),
    ("", False, 11),
    ("Please work independently — do not discuss your ratings with the other annotator until both sheets "
     "are complete and submitted.", True, 12),
]


def build_instructions_sheet(wb):
    ws = wb.create_sheet("Instructions")
    ws.column_dimensions["A"].width = 110
    for i, (text, bold, size) in enumerate(RUBRIC_TEXT, start=1):
        cell = ws.cell(row=i, column=1, value=text)
        cell.font = Font(bold=bold, size=size, name="Arial")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return ws


def build_ratings_sheet(wb, sheet_rows, fieldnames):
    ws = wb.create_sheet("Ratings")
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    editable_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    editable_cols = {"faithfulness", "answer_correctness_1to5", "kannada_fluency_1to5",
                      "context_relevance_1to5", "appropriate_refusal_yn", "annotator_notes"}

    for col_idx, name in enumerate(fieldnames, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = Font(bold=True, name="Arial")
        cell.fill = header_fill

    for row_idx, row in enumerate(sheet_rows, start=2):
        for col_idx, name in enumerate(fieldnames, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row.get(name, ""))
            cell.font = Font(name="Arial", size=11)
            if name in editable_cols:
                cell.fill = editable_fill

    # Column widths
    widths = {"question_id": 12, "difficulty": 10, "question": 45, "reference_answer": 45,
              "system_id": 8, "system_answer": 45, "context_relevance_1to5": 14,
              "faithfulness": 20, "answer_correctness_1to5": 16, "kannada_fluency_1to5": 14,
              "appropriate_refusal_yn": 16, "annotator_notes": 30}
    for col_idx, name in enumerate(fieldnames, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = widths.get(name, 15)

    n_rows = len(sheet_rows)
    last_row = n_rows + 1

    # Data validation: faithfulness dropdown
    faith_col = fieldnames.index("faithfulness") + 1
    dv_faith = DataValidation(type="list", formula1=f'"{",".join(FAITHFULNESS_OPTIONS)}"',
                               allow_blank=True, showDropDown=False)
    dv_faith.error = "Choose one of: " + ", ".join(FAITHFULNESS_OPTIONS)
    ws.add_data_validation(dv_faith)
    dv_faith.add(f"{get_column_letter(faith_col)}2:{get_column_letter(faith_col)}{last_row}")

    # Data validation: 1-5 integer scales
    for field in ["answer_correctness_1to5", "kannada_fluency_1to5", "context_relevance_1to5"]:
        col = fieldnames.index(field) + 1
        dv = DataValidation(type="whole", operator="between", formula1=1, formula2=5, allow_blank=True)
        dv.error = "Enter a whole number from 1 to 5"
        ws.add_data_validation(dv)
        dv.add(f"{get_column_letter(col)}2:{get_column_letter(col)}{last_row}")

    # Data validation: refusal y/n/na
    refusal_col = fieldnames.index("appropriate_refusal_yn") + 1
    dv_refusal = DataValidation(type="list", formula1=f'"{",".join(REFUSAL_OPTIONS)}"',
                                  allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv_refusal)
    dv_refusal.add(f"{get_column_letter(refusal_col)}2:{get_column_letter(refusal_col)}{last_row}")

    ws.freeze_panes = "A2"
    return ws


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

    sheet_rows, hidden_key_rows = [], []
    for qid in sampled_ids:
        rag_row, norag_row = rag_rows[qid], norag_rows[qid]
        entries = [
            {"system_id": "A", "answer": rag_row["answer"], "_true_system": "RAG"},
            {"system_id": "B", "answer": norag_row["answer"], "_true_system": "NoRAG"},
        ]
        random.shuffle(entries)
        for entry in entries:
            sheet_rows.append({
                "question_id": qid, "difficulty": rag_row.get("difficulty", ""),
                "question": rag_row["question"], "reference_answer": rag_row["reference"],
                "system_id": entry["system_id"], "system_answer": entry["answer"],
                "context_relevance_1to5": "", "faithfulness": "",
                "answer_correctness_1to5": "", "kannada_fluency_1to5": "",
                "appropriate_refusal_yn": "", "annotator_notes": "",
            })
            hidden_key_rows.append({"question_id": qid, "system_id": entry["system_id"],
                                     "true_system": entry["_true_system"]})

    fieldnames = ["question_id", "difficulty", "question", "reference_answer", "system_id",
                  "system_answer", "context_relevance_1to5", "faithfulness",
                  "answer_correctness_1to5", "kannada_fluency_1to5",
                  "appropriate_refusal_yn", "annotator_notes"]

    wb = Workbook()
    wb.remove(wb.active)
    build_instructions_sheet(wb)
    build_ratings_sheet(wb, sheet_rows, fieldnames)
    wb.save(out_path)

    key_path = out_path.rsplit(".", 1)[0] + "_ANSWER_KEY_DO_NOT_SHARE.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(hidden_key_rows, f, ensure_ascii=False, indent=2)

    print(f"\nWrote rater workbook -> {out_path}  ({len(sheet_rows)} rows, dropdowns enabled)")
    print(f"Wrote hidden answer key -> {key_path}  (keep this yourself, do NOT send to raters)")
    print("\nNext steps:")
    print(f"  1. Make two copies of {out_path}, e.g. rater1.xlsx and rater2.xlsx")
    print("  2. Send each copy plus a reminder to read the 'Instructions' tab first")
    print("  3. Once both come back, use the answer key + compute_rating_agreement.py to score them")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--rag_label", required=True)
    p.add_argument("--norag_label", required=True)
    p.add_argument("--n_per_tier", type=int, default=10)
    p.add_argument("--out", default="human_rating_workbook.xlsx")
    args = p.parse_args()
    run(args.results, args.rag_label, args.norag_label, args.n_per_tier, args.out)
