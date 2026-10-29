# Evaluation

Controlled evaluation of Kannada educational question answering: KN-BERT vs multilingual-E5 embeddings, dense vs hybrid (FAISS + BM25) retrieval, and RAG vs no retrieval, with `openai/gpt-oss-120b` and `openai/gpt-oss-20b` (Groq).

## Scope

The evaluated pipeline is embedding, dense or hybrid retrieval (top 5), and one generation call. The Query, Reranker and Retry agents, the Evaluation Agent's actions, and Edge-TTS output are part of the system but were not evaluated.

## Benchmark

`evaluation/kannada_ground_truth_76.json`: 76 questions (61 Kan-Science, 15 Kannada-EVS-5-Part1; 31 easy, 25 medium, 20 hard). The 25 original questions carry an audit status (12 verified, 8 partial, 5 unverified; `kannada_ground_truth_25_BACKFILLED.json`). The other 51 were written manually by annotators with subject and Kannada-language background and were not individually audited; 51 questions have annotated gold pages.

## Settings

Temperature 0.2; reasoning text excluded. Completion cap 800 tokens (120B) / 400 (20B), identical for RAG and No-RAG of the same generator. RAG context cap 3000 characters (120B) / 1800 (20B), with truncation. The No-RAG prompt differs from the RAG prompt. 120B and 20B are compared only within a generator.

## Metrics

ROUGE (`rouge-score`, whitespace tokenizer), a simplified token-overlap METEOR, character-level smoothed sentence BLEU, and BERT precision/recall/F1 (greedy matching, last layer, no IDF or rescaling) with the KN-BERT model (primary) or `bert-base-multilingual-cased` (independent). These measure similarity to the reference answer, not factual accuracy. Unanswered questions score 0 in full-denominator means; paired tests use questions answered by both systems.

## Statistics

Paired bootstrap (10,000 resamples, percentile CI, p floored at 1/10001) with Holm correction over a family of 16 comparisons (8 RAG vs same-generator No-RAG, 4 KN-BERT vs E5, 4 hybrid vs dense), applied within each difficulty stratum.

## Main results (BERT-F1, primary evaluator)

| Configuration | Overall | Answered only |
|---|---|---|
| KN-BERT Hybrid + 120B | 0.8330 | |
| E5 Hybrid + 120B | 0.8246 | |
| KN-BERT Hybrid + 20B | 0.8153 | 0.8261 |
| No-RAG + 120B | 0.7854 | 0.7958 |
| No-RAG + 20B | 0.7524 | |

- KN-BERT Hybrid + 120B vs No-RAG + 120B: paired gain +0.0376 [0.0155, 0.0590], Holm p = 0.012, n = 75.
- Independent evaluator: all eight RAG-vs-No-RAG gains (+0.033 to +0.065) are significant after correction.
- KN-BERT minus E5: all intervals include zero; all Holm p = 1.0 (primary and independent).
- Page-level retrieval (51 questions): Hit@5 0.840-0.882; MRR 0.624-0.751; embedders not separated.
- Human study (30 questions, 2 raters, blinded, evidence shown): no significant RAG vs No-RAG difference in faithfulness, correctness or fluency; faithfulness kappa 0.156 (4 categories). Disagreements were not adjudicated.

## Reproducing

```bash
python3 evaluation/compare_eval_v2.py --print_only evaluation/results/results_FINAL.json
python3 evaluation/verify_metrics.py evaluation/results/results_FINAL.json      # re-scores saved answers
python3 evaluation/significance_test.py --help                                   # paired tests (--family)
python3 evaluation/independent_bertscore.py --help
python3 evaluation/retrieval_eval.py --help
python3 evaluation/split_by_support.py --help
python3 evaluation/score_human_ratings.py --rater1 R1.xlsx --rater2 R2.xlsx \
    --key evaluation/human_study/answer_key.json
```

`evaluation/results/results_FINAL.json` (sha256 `5a39c432...e22f`) holds every per-question prediction and score; all tables derive from it. Anonymised human ratings: `evaluation/human_study/ratings_release.csv`.
