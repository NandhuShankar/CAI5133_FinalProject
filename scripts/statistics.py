"""
#Nandhu Shankar
program for statistical tests, claude assisted in coding this
Tests:
  1. Overall chi-square across all 3 tiers × 6 macro-emotions
  2. Pairwise chi-square (Tier0 vs Tier1, Tier0 vs Tier2, Tier1 vs Tier2)
     with Bonferroni correction
  3. Cramér's V effect size for each comparison
  4. Per-topic chi-square for the value map topics (Tier1 vs Tier2)

Input  : emotions_tagged.csv
Output : statistics_report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from itertools import combinations


def cramers_v(contingency_table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n    = contingency_table.sum()
    k    = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def interpret_v(v: float) -> str:
    if v < 0.1:  return "negligible"
    if v < 0.2:  return "small"
    if v < 0.3:  return "medium"
    return "large"


def chi2_summary(label: str, table: np.ndarray,
                 bonferroni_k: int = 1) -> dict:
    chi2, p, dof, _ = chi2_contingency(table)
    p_corrected      = min(p * bonferroni_k, 1.0)
    v                = cramers_v(table)
    sig              = "***" if p_corrected < 0.001 else \
                       "**"  if p_corrected < 0.01  else \
                       "*"   if p_corrected < 0.05  else "ns"
    return {
        "comparison":    label,
        "chi2":          round(chi2, 2),
        "dof":           dof,
        "p_raw":         p,
        "p_corrected":   p_corrected,
        "sig":           sig,
        "cramers_v":     round(v, 4),
        "effect":        interpret_v(v),
        "n":             int(table.sum()),
    }


def print_result(r: dict, f=None) -> None:
    lines = [
        f"\n  Comparison : {r['comparison']}",
        f"  N          : {r['n']:,}",
        f"  χ²         : {r['chi2']:.2f}  (df={r['dof']})",
        f"  p (raw)    : {r['p_raw']:.2e}",
        f"  p (corr.)  : {r['p_corrected']:.2e}  {r['sig']}",
        f"  Cramér's V : {r['cramers_v']:.4f}  [{r['effect']} effect]",
    ]
    for line in lines:
        print(line)
        if f:
            f.write(line + "\n")


INPUT_CSV  = "comments_full_analysis.csv"
OUTPUT_TXT = "statistics_report.txt"

TIER_ORDER = ["tier0_baseline", "tier1_general_tech", "tier2_niche_tech"]
TIER_SHORT = {
    "tier0_baseline":     "Tier0",
    "tier1_general_tech": "Tier1",
    "tier2_niche_tech":   "Tier2",
}

if not Path(INPUT_CSV).exists():
    print(f"[ERROR] {INPUT_CSV} not found. Run analyze_comments.py first.")
    exit(1)

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} comments")

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:

    header = "=" * 70 + "\nSTATISTICAL VALIDATION REPORT\n" + "=" * 70
    print(header); f.write(header + "\n")

    # ── 1. Contingency table ──────────────────────────────────────────────────
    section = "\n── EMOTION × TIER CONTINGENCY TABLE (counts) ───────────────────────"
    print(section); f.write(section + "\n")

    ct = pd.crosstab(df["tier"], df["macro_emotion"])
    ct = ct.reindex(TIER_ORDER)                         # consistent row order
    print(ct.to_string()); f.write(ct.to_string() + "\n")

    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    pct_str = "\n  (Row percentages):\n" + ct_pct.to_string()
    print(pct_str); f.write(pct_str + "\n")

    # ── 2. Overall chi-square (all 3 tiers) ───────────────────────────────────
    section = "\n── TEST 1: OVERALL CHI-SQUARE (all 3 tiers) ────────────────────────"
    print(section); f.write(section + "\n")

    r_overall = chi2_summary("All tiers vs all emotions", ct.values)
    print_result(r_overall, f)

    note = ("\n  Interpretation: tests whether emotion distributions differ\n"
            "  significantly across the three audience tiers overall.")
    print(note); f.write(note + "\n")

    # ── 3. Pairwise chi-square with Bonferroni ────────────────────────────────
    section = "\n── TEST 2: PAIRWISE CHI-SQUARE (Bonferroni k=3) ────────────────────"
    print(section); f.write(section + "\n")

    pairs = list(combinations(TIER_ORDER, 2))
    pairwise_results = []
    for t1, t2 in pairs:
        sub_ct = ct.loc[[t1, t2]].values
        label  = f"{TIER_SHORT[t1]} vs {TIER_SHORT[t2]}"
        r      = chi2_summary(label, sub_ct, bonferroni_k=3)
        pairwise_results.append(r)
        print_result(r, f)

    # ── 4. Cramér's V summary table ───────────────────────────────────────────
    section = "\n── EFFECT SIZE SUMMARY (Cramér's V) ────────────────────────────────"
    print(section); f.write(section + "\n")

    all_results = [r_overall] + pairwise_results
    v_df = pd.DataFrame([{
        "Comparison": r["comparison"],
        "Cramér's V": r["cramers_v"],
        "Effect":     r["effect"],
        "p (corr.)":  f"{r['p_corrected']:.2e}",
        "Sig":        r["sig"],
    } for r in all_results])

    v_str = v_df.to_string(index=False)
    print(v_str); f.write(v_str + "\n")

    legend = ("\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant"
              "\n  Effect size : negligible<0.1  small<0.2  medium<0.3  large≥0.3")
    print(legend); f.write(legend + "\n")

    # ── 5. Per-emotion contribution (standardised residuals) ──────────────────
    section = "\n── TEST 3: STANDARDISED RESIDUALS ──────────────────────────────────"
    note    = ("  Values > ±2.0 indicate a cell contributes significantly to χ².\n"
               "  Use these to identify WHICH emotions drive tier differences.")
    print(section); f.write(section + "\n")
    print(note);    f.write(note + "\n")

    chi2, p, dof, expected = chi2_contingency(ct.values)
    residuals = (ct.values - expected) / np.sqrt(expected)
    res_df    = pd.DataFrame(residuals,
                             index=ct.index,
                             columns=ct.columns).round(2)
    res_str   = res_df.to_string()
    print(res_str); f.write(res_str + "\n")

    highlight = "\n  Cells with |residual| > 2.0 (significant drivers):"
    print(highlight); f.write(highlight + "\n")

    for tier in res_df.index:
        for emotion in res_df.columns:
            val = res_df.loc[tier, emotion]
            if abs(val) >= 2.0:
                direction = "HIGHER than expected" if val > 0 else "LOWER than expected"
                line = f"    {TIER_SHORT[tier]:6s} × {emotion:<26s}: {val:+.2f}  ({direction})"
                print(line); f.write(line + "\n")

    # ── 6. Per-topic chi-square (Tier1 vs Tier2 value map) ───────────────────
    section = "\n── TEST 4: PER-TOPIC CHI-SQUARE (Tier1 vs Tier2) ───────────────────"
    note    = "  Tests whether emotion distributions differ per topic across tiers."
    print(section); f.write(section + "\n")
    print(note);    f.write(note + "\n")

    tech_df = df[df["tier"].isin(["tier1_general_tech", "tier2_niche_tech"])].copy()
    tech_df = tech_df[tech_df["topic_label"].notna()]
    tech_df = tech_df[tech_df["topic_label"] != "outlier"]

    if "topic_label" not in df.columns or tech_df.empty:
        msg = "\n  [SKIP] topic_label column not found — run analyze_comments.py first."
        print(msg); f.write(msg + "\n")
    else:
        # Only topics that appear in BOTH tiers
        tier1_topics = set(tech_df[tech_df["tier"]=="tier1_general_tech"]["topic_label"])
        tier2_topics = set(tech_df[tech_df["tier"]=="tier2_niche_tech"]["topic_label"])
        shared = tier1_topics & tier2_topics

        if not shared:
            msg = "\n  No shared topics between Tier1 and Tier2 — reporting per-tier only."
            print(msg); f.write(msg + "\n")

        # Report per-tier topic emotion distributions with chi-square vs overall
        for tier in ["tier1_general_tech", "tier2_niche_tech"]:
            t_df     = tech_df[tech_df["tier"] == tier]
            topics   = t_df["topic_label"].unique()
            sub_head = f"\n  [{TIER_SHORT[tier]}] — chi-square per topic vs tier overall:"
            print(sub_head); f.write(sub_head + "\n")

            tier_overall = t_df["macro_emotion"].value_counts().sort_index()

            for topic in sorted(topics,
                                key=lambda x: -len(t_df[t_df["topic_label"]==x])):
                grp = t_df[t_df["topic_label"] == topic]
                if len(grp) < 20:
                    continue   # too small to test

                # contingency: topic row vs rest-of-tier row
                topic_counts = grp["macro_emotion"].value_counts()
                rest_counts  = (tier_overall - topic_counts).clip(lower=0)

                # Align on same emotion columns
                all_emotions = sorted(set(topic_counts.index) | set(rest_counts.index))
                row_topic = [topic_counts.get(e, 0) for e in all_emotions]
                row_rest  = [rest_counts.get(e,  0) for e in all_emotions]
                table     = np.array([row_topic, row_rest])

                if table.min() == 0 or table.shape[1] < 2:
                    continue

                try:
                    r = chi2_summary(
                        f"  topic '{topic[:30]}' vs rest of {TIER_SHORT[tier]}",
                        table
                    )
                    sig_marker = r["sig"]
                    line = (f"    {topic[:32]:<33} n={len(grp):4d}  "
                            f"χ²={r['chi2']:7.2f}  "
                            f"V={r['cramers_v']:.3f} [{r['effect']:10s}]  "
                            f"{sig_marker}")
                    print(line); f.write(line + "\n")
                except Exception:
                    continue

    # ── 7. Descriptive stats ──────────────────────────────────────────────────
    section = "\n── DESCRIPTIVE STATS (for methods section) ────────────────────────"
    print(section); f.write(section + "\n")

    desc = df.groupby("tier").agg(
        N              = ("text_clean", "count"),
        avg_words      = ("word_count",  "mean"),
        median_words   = ("word_count",  "median"),
        avg_chars      = ("char_count",  "mean"),
        pct_emoji      = ("has_emoji",   "mean"),
    ).round(2)
    desc["pct_emoji"] = (desc["pct_emoji"] * 100).round(1)
    desc_str = desc.to_string()
    print(desc_str); f.write(desc_str + "\n")

    # ── 8. Per-claim statistical backing ─────────────────────────────────────
    # Every percentage claim backed by a z-test for proportions.
    # Tests whether the observed proportion in a tier significantly differs
    # from the overall population proportion (two-sided, alpha=0.05).
    # ─────────────────────────────────────────────────────────────────────────
    section = "\n── TEST 5: PER-CLAIM PROPORTION Z-TESTS ────────────────────────────"
    note    = ("  Each claim: 'Tier X has Y% of emotion Z'\n"
               "  H0: proportion in tier = proportion in full dataset\n"
               "  Test: two-proportion z-test (tier vs rest-of-dataset)\n"
               "  Bonferroni corrected for number of claims tested.")
    print(section); f.write(section + "\n")
    print(note);    f.write(note + "\n")

    from scipy.stats import norm as scipy_norm

    # Define the key claims from your presentation
    claims = [
        # (description, tier, emotion)
        ("Tier0 amusement elevated",        "tier0_baseline",     "amusement/joy"),
        ("Tier1 curiosity elevated",        "tier1_general_tech", "curiosity/interest"),
        ("Tier1 negative elevated",         "tier1_general_tech", "negative/critical"),
        ("Tier2 admiration elevated",       "tier2_niche_tech",   "admiration/approval"),
        ("Tier0 curiosity suppressed",      "tier0_baseline",     "curiosity/interest"),
        ("Tier1 amusement suppressed",      "tier1_general_tech", "amusement/joy"),
        ("Tier2 amusement suppressed",      "tier2_niche_tech",   "amusement/joy"),
        ("Tier2 concern suppressed",        "tier2_niche_tech",   "concern/disappointment"),
    ]

    bonferroni_k = len(claims)
    header_line  = (f"  {'Claim':<35} {'Tier%':>6} {'Pop%':>6} "
                    f"{'z':>7} {'p_corr':>10} {'sig':>4}")
    print(header_line); f.write(header_line + "\n")
    print("  " + "-"*78); f.write("  " + "-"*78 + "\n")

    for desc, tier, emotion in claims:
        tier_df  = df[df["tier"] == tier]
        n_tier   = len(tier_df)
        x_tier   = (tier_df["macro_emotion"] == emotion).sum()
        p_tier   = x_tier / n_tier

        n_pop    = len(df)
        x_pop    = (df["macro_emotion"] == emotion).sum()
        p_pop    = x_pop / n_pop

        # Two-proportion z-test: tier proportion vs population proportion
        p_pool   = (x_tier + x_pop) / (n_tier + n_pop)
        se       = ((p_pool * (1 - p_pool)) * (1/n_tier + 1/n_pop)) ** 0.5
        z        = (p_tier - p_pop) / se if se > 0 else 0
        p_raw    = 2 * (1 - scipy_norm.cdf(abs(z)))
        p_corr   = min(p_raw * bonferroni_k, 1.0)
        sig      = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"

        line = (f"  {desc:<35} {p_tier*100:>5.1f}% {p_pop*100:>5.1f}% "
                f"{z:>+7.2f} {p_corr:>10.2e} {sig:>4}")
        print(line); f.write(line + "\n")

    legend2 = ("\n  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant"
               "\n  Bonferroni corrected for " + str(bonferroni_k) + " simultaneous claims"
               "\n  Tier% = proportion in that tier, Pop% = proportion in full dataset")
    print(legend2); f.write(legend2 + "\n")

    # ── 9. Word count statistical test (Kruskal-Wallis) ──────────────────────
    # Non-parametric test for word count differences across tiers
    # (word counts are not normally distributed so t-test is inappropriate)
    section = "\n── TEST 6: WORD COUNT DIFFERENCES (Kruskal-Wallis) ─────────────────"
    note    = "  Non-parametric test: are word count distributions different across tiers?"
    print(section); f.write(section + "\n")
    print(note);    f.write(note + "\n")

    from scipy.stats import kruskal, mannwhitneyu

    groups   = [df[df["tier"]==t]["word_count"].dropna().values for t in TIER_ORDER]
    stat, p  = kruskal(*groups)
    sig      = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    kw_line  = f"  Kruskal-Wallis H = {stat:.2f},  p = {p:.2e}  {sig}"
    print(kw_line); f.write(kw_line + "\n")

    # Pairwise Mann-Whitney U with Bonferroni
    print("\n  Pairwise Mann-Whitney U (Bonferroni k=3):")
    f.write("\n  Pairwise Mann-Whitney U (Bonferroni k=3):\n")
    for t1, t2 in combinations(TIER_ORDER, 2):
        g1   = df[df["tier"]==t1]["word_count"].dropna().values
        g2   = df[df["tier"]==t2]["word_count"].dropna().values
        u, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")
        p_corr   = min(p_raw * 3, 1.0)
        sig      = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        med1, med2 = np.median(g1), np.median(g2)
        mw_line  = (f"  {TIER_SHORT[t1]} vs {TIER_SHORT[t2]:<8}: "
                    f"U={u:.0f},  p={p_corr:.2e}  {sig}  "
                    f"(medians: {med1:.0f} vs {med2:.0f} words)")
        print(mw_line); f.write(mw_line + "\n")
    section = "\n── TEST 7: EMOJI USAGE DIFFERENCES (Chi-square) ────────────────────"
    print(section); f.write(section + "\n")

    emoji_ct = pd.crosstab(df["tier"], df["has_emoji"]).reindex(TIER_ORDER)
    r = chi2_summary("Emoji usage across tiers", emoji_ct.values)
    print_result(r, f)

    footer = f"\n{'='*70}\nSaved → {OUTPUT_TXT}\n{'='*70}"
    print(footer); f.write(footer + "\n")
