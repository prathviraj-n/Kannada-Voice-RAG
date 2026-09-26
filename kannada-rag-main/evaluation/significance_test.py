"""
evaluation/significance_test.py

Statistical significance testing for paired RAG system comparisons.

Design rationale (see accompanying reviewer-response notes):
  - Same question set answered by every system  -> PAIRED tests, not
    independent-sample tests.
  - Small n (25, growing to ~150-250)            -> nonparametric primary
    test (Wilcoxon signed-rank) + paired bootstrap CI, no normality assumed.
  - Coverage (answered/skipped) is binary         -> McNemar's test, not
    a test built for continuous scores.
  - Many comparisons run in one study             -> Holm-Bonferroni
    correction applied across the declared comparison set.

USAGE:
  python evaluation/significance_test.py \
      --results evaluation/results_v2_20260910_1200.json \
      --metric bert_f1 \
      --pairs "kannada-bert|hybrid|llama-3.1-8b-instant:e5|hybrid|llama-3.1-8b-instant" \
              "kannada-bert|hybrid|llama-3.1-8b-instant:NoRAG|llama-3.1-8b-instant"

Each --pairs entry is "labelA:labelB" using the exact combo labels that
appear as keys under output["combos"] / output["no_rag"] in your results
JSON (no_rag entries are already prefixed "NoRAG|<model>" by your eval script).
"""

import json, argparse, itertools
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)  # fixed seed -> reproducible p-values/CIs


# ─────────────────────────────────────────────────────────────
#  Loading & pairing
# ─────────────────────────────────────────────────────────────

def load_all_labels(results: Dict) -> Dict[str, List[Dict]]:
    out = dict(results.get("combos", {}))
    out.update(results.get("no_rag", {}))
    return out


def paired_scores(rows_a: List[Dict], rows_b: List[Dict], metric: str,
                   difficulty: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Aligns two systems' per-question rows by question id.
    Excludes any question skipped by EITHER system for this metric —
    report how many were excluded; don't silently zero-fill for a
    significance test (zero-filling belongs in the coverage-honest mean,
    not in a paired difference test, since a skip isn't a real score).
    """
    a_by_id = {r["id"]: r for r in rows_a}
    b_by_id = {r["id"]: r for r in rows_b}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    xs, ys, used_ids = [], [], []
    for qid in common_ids:
        ra, rb = a_by_id[qid], b_by_id[qid]
        if difficulty and ra.get("difficulty") != difficulty:
            continue
        if ra.get("skipped") or rb.get("skipped"):
            continue
        xs.append(ra.get(metric, 0.0))
        ys.append(rb.get(metric, 0.0))
        used_ids.append(qid)
    return np.array(xs), np.array(ys), used_ids


# ─────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────

def paired_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP):
    """
    Resamples question INDICES (not scores independently) with replacement,
    preserving the pairing. Returns mean diff, 95% CI, and a two-sided
    bootstrap p-value (proportion of resamples where the sign flips).
    """
    n = len(x)
    diffs = x - y
    obs_mean = diffs.mean()

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    # two-sided p: proportion of bootstrap means on the opposite side of 0
    # from the observed mean, doubled (standard percentile-bootstrap p-value)
    if obs_mean >= 0:
        p = 2 * min((boot_means <= 0).mean(), 0.5)
    else:
        p = 2 * min((boot_means >= 0).mean(), 0.5)
    return {"mean_diff": round(float(obs_mean), 4),
            "ci95": (round(float(ci_lo), 4), round(float(ci_hi), 4)),
            "p_bootstrap": round(float(p), 4)}


def wilcoxon_test(x: np.ndarray, y: np.ndarray):
    diffs = x - y
    if np.allclose(diffs, 0):
        return {"p_wilcoxon": 1.0, "statistic": 0.0, "rank_biserial": 0.0}
    try:
        stat, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
    except ValueError as e:
        return {"p_wilcoxon": None, "statistic": None, "error": str(e)}
    n_nonzero = np.sum(diffs != 0)
    rank_biserial = stat / (n_nonzero * (n_nonzero + 1) / 2) if n_nonzero else 0.0
    return {"p_wilcoxon": round(float(p), 4),
            "statistic": round(float(stat), 2),
            "rank_biserial": round(float(rank_biserial), 4)}


def mcnemar_coverage(rows_a: List[Dict], rows_b: List[Dict]):
    """
    McNemar's test on paired answered/skipped outcomes.
    Use this for coverage comparisons, NOT continuous-score tests.
    """
    a_by_id = {r["id"]: not r.get("skipped", False) for r in rows_a}
    b_by_id = {r["id"]: not r.get("skipped", False) for r in rows_b}
    common = sorted(set(a_by_id) & set(b_by_id))
    b01 = sum(1 for q in common if a_by_id[q] and not b_by_id[q])   # A answered, B skipped
    b10 = sum(1 for q in common if not a_by_id[q] and b_by_id[q])   # A skipped, B answered
    n_disc = b01 + b10
    if n_disc == 0:
        return {"p_mcnemar": 1.0, "discordant_pairs": 0}
    # exact binomial McNemar for small n (more appropriate than chi-square when n_disc is small)
    p = stats.binomtest(min(b01, b10), n_disc, 0.5).pvalue
    return {"p_mcnemar": round(float(p), 4), "discordant_pairs": n_disc,
            "a_answered_b_skipped": b01, "a_skipped_b_answered": b10}


# ─────────────────────────────────────────────────────────────
#  Multiple comparison correction
# ─────────────────────────────────────────────────────────────

def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Returns Holm-Bonferroni adjusted p-values, same order as input."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def run(results_path: str, metric: str, pairs: List[str], by_difficulty: bool):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    all_labels = load_all_labels(results)

    difficulties = ["overall", "easy", "medium", "hard"] if by_difficulty else ["overall"]
    records = []

    for pair_str in pairs:
        label_a, label_b = pair_str.split(":")
        if label_a not in all_labels or label_b not in all_labels:
            print(f"  ⚠️  Skipping '{pair_str}' — label not found in results.")
            continue
        rows_a, rows_b = all_labels[label_a], all_labels[label_b]

        for diff in difficulties:
            d = None if diff == "overall" else diff
            x, y, used_ids = paired_scores(rows_a, rows_b, metric, d)
            if len(x) < 5:
                print(f"  ⚠️  {label_a} vs {label_b} [{diff}]: only {len(x)} paired, "
                      f"non-skipped questions — too few for a reliable test, skipping.")
                continue

            boot = paired_bootstrap(x, y)
            wil = wilcoxon_test(x, y)
            mcn = mcnemar_coverage(rows_a, rows_b)

            records.append({
                "comparison": f"{label_a} vs {label_b}",
                "difficulty": diff,
                "metric": metric,
                "n_paired": len(x),
                **boot, **wil, **mcn,
            })

    # Holm-Bonferroni correction across all bootstrap p-values in this run
    pvals = [r["p_bootstrap"] for r in records]
    if pvals:
        adj = holm_bonferroni(pvals)
        for r, a in zip(records, adj):
            r["p_bootstrap_holm"] = round(a, 4)

    print(f"\n{'='*100}")
    print(f"  Significance test results — metric: {metric}")
    print(f"{'='*100}")
    for r in records:
        sig = "significant (Holm-corrected)" if r.get("p_bootstrap_holm", 1.0) < 0.05 else "not significant"
        print(f"\n  {r['comparison']}  [{r['difficulty']}]  n={r['n_paired']}")
        print(f"    mean diff = {r['mean_diff']:+.4f}   95% CI = {r['ci95']}")
        print(f"    bootstrap p = {r['p_bootstrap']:.4f}   Holm-adjusted p = {r.get('p_bootstrap_holm','-')}"
              f"   -> {sig}")
        print(f"    Wilcoxon p = {r.get('p_wilcoxon','-')}   rank-biserial = {r.get('rank_biserial','-')}")
        print(f"    coverage McNemar p = {r.get('p_mcnemar','-')}  "
              f"(discordant pairs = {r.get('discordant_pairs','-')})")

    out_path = results_path.replace(".json", f"_sigtest_{metric}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Full results -> {out_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--metric", default="bert_f1")
    p.add_argument("--pairs", nargs="+", required=True,
                   help='Each entry: "labelA:labelB" using exact keys from results JSON')
    p.add_argument("--by_difficulty", action="store_true",
                   help="Also break down easy/medium/hard, not just overall")
    args = p.parse_args()
    run(args.results, args.metric, args.pairs, args.by_difficulty)
