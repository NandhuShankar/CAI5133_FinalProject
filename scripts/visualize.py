#Nandhu Shankar
#charts for presentation
#claude assisted with the crating these charts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path

Path("charts").mkdir(exist_ok=True)

COLORS = {
    "tier0": "#AAAAAA",  
    "tier1": "#444444",   
    "tier2": "#111111",  
}
TIER_LABELS = {
    "tier0_baseline":     "Tier 0 – Baseline",
    "tier1_general_tech": "Tier 1 – General Tech",
    "tier2_niche_tech":   "Tier 2 – Niche Tech",
}
TIER_ORDER = ["tier0_baseline", "tier1_general_tech", "tier2_niche_tech"]
COLOR_LIST = [COLORS["tier0"], COLORS["tier1"], COLORS["tier2"]]

EMOTION_COLORS = {
    "admiration/approval":       "#2c7bb6",
    "amusement/joy":             "#f4a261",
    "concern/disappointment":    "#e76f51",
    "curiosity/interest":        "#57cc99",
    "negative/critical":         "#c9184a",
    "neutral/analytical":        "#adb5bd",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

def save(fig, name):
    path = f"charts/{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")

df = pd.read_csv("comments_full_analysis.csv", low_memory=False)
print(f"  {len(df):,} comments loaded")


print("\nChart 1: Emotion % by Tier...")

emotions = ["admiration/approval","amusement/joy","concern/disappointment",
            "curiosity/interest","negative/critical","neutral/analytical"]
short_emotions = ["Admiration","Amusement","Concern","Curiosity","Negative","Neutral"]

ct = pd.crosstab(df["tier"], df["macro_emotion"])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct = ct_pct.reindex(TIER_ORDER)[emotions]

x     = np.arange(len(emotions))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))

for i, (tier, color) in enumerate(zip(TIER_ORDER, COLOR_LIST)):
    bars = ax.bar(x + i*width, ct_pct.loc[tier], width,
                  label=TIER_LABELS[tier], color=color, alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        if h >= 3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(short_emotions, fontsize=11)
ax.set_ylabel("% of Comments")
ax.set_title("Emotion % by Tier  (N=36,876)", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, 75)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fig.tight_layout()
save(fig, "chart1_emotion_by_tier")

print("Chart 2: Standardised Residuals heatmap")

residuals = {
    "tier0_baseline":     [-1.22, 13.25,  1.89, -4.93, -3.91, -2.35],
    "tier1_general_tech": [-1.29, -9.74,  0.86,  6.44,  5.22,  0.53],
    "tier2_niche_tech":   [ 2.22, -6.42, -2.60,  0.29,  0.14,  2.03],
}
res_df = pd.DataFrame(residuals, index=short_emotions).T
res_df.index = [TIER_LABELS[t] for t in TIER_ORDER]

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(res_df.values, cmap="RdYlGn", aspect="auto",
               vmin=-10, vmax=10)

ax.set_xticks(range(len(short_emotions)))
ax.set_xticklabels(short_emotions, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(res_df.index, fontsize=11)
ax.set_title("Standardised Residuals  (|value| > 2.0 = significant driver)",
             fontsize=12, fontweight="bold", pad=12)

for i in range(3):
    for j in range(6):
        val = res_df.values[i, j]
        color = "white" if abs(val) > 6 else "black"
        marker = "▲" if val >= 2.0 else ("▼" if val <= -2.0 else "")
        ax.text(j, i, f"{val:+.1f}{marker}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold" if abs(val) >= 2 else "normal")

plt.colorbar(im, ax=ax, shrink=0.8, label="Standardised Residual")
ax.grid(False)
fig.tight_layout()
save(fig, "chart2_residuals_heatmap")

print("Chart 3: Word count distribution")

fig, ax = plt.subplots(figsize=(9, 5))
data_by_tier = [df[df["tier"]==t]["word_count"].clip(upper=100).dropna().values
                for t in TIER_ORDER]

bp = ax.boxplot(data_by_tier, patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2))

for patch, color in zip(bp["boxes"], COLOR_LIST):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=10)
ax.set_ylabel("Word Count (capped at 100)")
ax.set_title("Comment Length by Tier\n(proxy for audience engagement depth)",
             fontsize=12, fontweight="bold", pad=10)

# Annotate medians
medians = [10.91, 30.10, 23.96]   # avg word counts from descriptive stats
for i, (med, color) in enumerate(zip(medians, COLOR_LIST)):
    ax.text(i+1, bp["medians"][i].get_ydata()[0] + 1.5,
            f"avg {med:.0f}w", ha="center", fontsize=9, color="black")

fig.tight_layout()
save(fig, "chart3_wordcount_distribution")

def _plot_value_map(topic_data: dict, title: str, filename: str,
                    neutral_pct: float = 60.0):
    NON_NEUTRAL = ["admiration/approval","amusement/joy","curiosity/interest",
                   "negative/critical","concern/disappointment"]
    SHORT_E     = ["Admiration","Amusement","Curiosity","Negative","Concern"]
    BAR_COLORS  = ["#2c7bb6","#f4a261","#57cc99","#c9184a","#e76f51"]

    topics = list(topic_data.keys())
    data   = np.array([[topic_data[t].get(e, 0) for e in NON_NEUTRAL]
                        for t in topics])

    x     = np.arange(len(topics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))

    for j, (short, color) in enumerate(zip(SHORT_E, BAR_COLORS)):
        bars = ax.bar(x + j*width, data[:, j], width,
                      label=short, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h >= 5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(topics, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("% of Topic Comments")
    ax.set_title(f"{title}\n(neutral/analytical ~{neutral_pct:.0f}% excluded)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 30)
    fig.tight_layout()
    save(fig, filename)

def extract_topic_emotion(df: pd.DataFrame, tier: str,
                          top_n: int = 5) -> tuple:
    """
    For a given tier, compute the % of each non-neutral macro_emotion
    per topic_label. Returns (topic_dict, avg_neutral_pct).
    """
    NON_NEUTRAL = ["admiration/approval","amusement/joy","curiosity/interest",
                   "negative/critical","concern/disappointment"]

    t_df = df[(df["tier"] == tier) &
              (df["topic_label"].notna()) &
              (df["topic_label"] != "outlier")].copy()

    # Pick top_n topics by comment count
    top_topics = (t_df.groupby("topic_label").size()
                      .sort_values(ascending=False)
                      .head(top_n).index.tolist())

    topic_dict   = {}
    neutral_pcts = []

    for topic in top_topics:
        grp    = t_df[t_df["topic_label"] == topic]
        total  = len(grp)
        counts = grp["macro_emotion"].value_counts()
        pcts   = (counts / total * 100).round(1)

        neutral_pcts.append(pcts.get("neutral/analytical", 0))

        # Shorten label to 3 keywords max
        short = " / ".join(topic.split(" / ")[:3])
        topic_dict[short] = {e: pcts.get(e, 0.0) for e in NON_NEUTRAL}

    avg_neutral = float(np.mean(neutral_pcts)) if neutral_pcts else 60.0
    return topic_dict, avg_neutral


print("Chart 4: Value map Tier 1 (from data)...")
t1_data, t1_neutral = extract_topic_emotion(df, "tier1_general_tech", top_n=5)
_plot_value_map(t1_data,
                "Tier 1 (General Tech) — Topic × Emotion Value Map",
                "chart4_valuemap_tier1",
                neutral_pct=t1_neutral)

print("Chart 5: Value map Tier 2 (from data)...")
t2_data, t2_neutral = extract_topic_emotion(df, "tier2_niche_tech", top_n=5)
_plot_value_map(t2_data,
                "Tier 2 (Niche Tech) — Topic × Emotion Value Map",
                "chart5_valuemap_tier2",
                neutral_pct=t2_neutral)

print("Chart 6: Emoji usage & word count...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Emoji %
emoji_vals = [19.0, 9.0, 5.0]
bars = ax1.bar([TIER_LABELS[t].split("–")[1].strip() for t in TIER_ORDER],
               emoji_vals, color=COLOR_LIST, alpha=0.85, width=0.5)
for bar, val in zip(bars, emoji_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_title("Emoji Usage by Tier\n(% of comments containing emoji)",
              fontsize=11, fontweight="bold")
ax1.set_ylabel("%")
ax1.set_ylim(0, 25)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

# Avg word count
word_vals = [10.91, 30.10, 23.96]
bars2 = ax2.bar([TIER_LABELS[t].split("–")[1].strip() for t in TIER_ORDER],
                word_vals, color=COLOR_LIST, alpha=0.85, width=0.5)
for bar, val in zip(bars2, word_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_title("Average Word Count by Tier\n(proxy for engagement depth)",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Avg Words per Comment")
ax2.set_ylim(0, 38)

fig.tight_layout(pad=3)
save(fig, "chart6_emoji_wordcount")

print("Chart 7: Effect size summary...")

comparisons = ["All Tiers\n(overall)", "Tier0\nvs Tier1", "Tier0\nvs Tier2", "Tier1\nvs Tier2"]
v_values    = [0.0780, 0.1211, 0.0878, 0.0566]
sig_labels  = ["p=3e-90 ***", "p=1.1e-69 ***", "p=7e-46 ***", "p=3.1e-13 ***"]
bar_colors  = ["#444444", "#2c7bb6", "#57cc99", "#f4a261"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(comparisons, v_values, color=bar_colors, alpha=0.85, width=0.5)

# threshold lines
ax.axhline(0.1, color="red",    linestyle="--", linewidth=1.2, alpha=0.6, label="Small (0.1)")
ax.axhline(0.2, color="orange", linestyle="--", linewidth=1.2, alpha=0.6, label="Medium (0.2)")
ax.axhline(0.3, color="green",  linestyle="--", linewidth=1.2, alpha=0.6, label="Large (0.3)")

for bar, val, sig in zip(bars, v_values, sig_labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"V={val:.3f}\n{sig}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Cramér's V")
ax.set_title("Effect Size Summary (Cramér's V)\nAll comparisons significant at p<0.001",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 0.20)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
save(fig, "chart7_cramers_v")


print("\nAll charts saved to charts/ folder.")
print("Files:")
for f in sorted(Path("charts").glob("*.png")):
    print(f"  {f}")
