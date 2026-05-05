#Nandhu Shankar
#Downsamples comments to ensure balanced dataset

import pandas as pd
from pathlib import Path

try:
    from langdetect import detect, LangDetectException
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect", "-q"])
    from langdetect import detect, LangDetectException

INPUT_CSV        = "comments_unified.csv"
OUTPUT_CSV       = "comments_balanced.csv"

TIER_TARGETS = {
    "tier0_baseline":     15000,
    "tier1_general_tech": None, 
    "tier2_niche_tech":   None,  
}

RANDOM_SEED = 42

#functions to filter out non-english comments
def fast_is_english(text: str) -> str:

    import re
    if not isinstance(text, str) or len(text) == 0:
        return "unknown"
    emoji_re = re.compile("[\U0001F300-\U0001FAFF"
                          "\U00002600-\U000027BF"
                          "\U0001F900-\U0001F9FF]+", flags=re.UNICODE)
    stripped = emoji_re.sub("", text)
    if len(stripped) == 0:
        return "en"         

    non_ascii = sum(1 for c in stripped if ord(c) > 127)
    ratio     = non_ascii / len(stripped)

    if ratio > 0.15:       
        return "non-en"
    if ratio < 0.05:         
        return "en"

    try:
        return detect(stripped)
    except LangDetectException:
        return "en"          


def filter_english(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    total = len(df)
    print(f"  Detecting language for {total:,} comments (fast two-stage filter)...")

    df = df.copy()
    df["lang"] = df[text_col].apply(fast_is_english)

    english_df = df[df["lang"] == "en"].reset_index(drop=True)
    dropped    = total - len(english_df)
    print(f"  Kept {len(english_df):,} English comments, dropped {dropped:,} "
          f"({dropped/total*100:.1f}%) non-English.")
    return english_df

#downsampling

def downsample(df: pd.DataFrame, targets: dict, seed: int = 42) -> pd.DataFrame:
    parts = []
    for tier, group in df.groupby("tier"):
        target = targets.get(tier)
        if target is None or len(group) <= target:
            parts.append(group)
            print(f"  {tier:25s}: keeping all {len(group):,} comments")
        else:
            sampled = group.sample(n=target, random_state=seed)
            parts.append(sampled)
            print(f"  {tier:25s}: downsampled {len(group):,} → {target:,}")
    return pd.concat(parts, ignore_index=True)


def summarise(df: pd.DataFrame) -> None:
    summary = df.groupby("tier").agg(
        total_comments    = ("text_clean", "count"),
        avg_word_count    = ("word_count", "mean"),
        median_word_count = ("word_count", "median"),
        avg_char_count    = ("char_count", "mean"),
        pct_has_emoji     = ("has_emoji",  "mean"),
        unique_authors    = ("author",     "nunique"),
    ).round(2)

    summary["pct_has_emoji"] = (summary["pct_has_emoji"] * 100).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    if not Path(INPUT_CSV).exists():
        print(f"[ERROR] {INPUT_CSV} not found.")
        exit(1)

    print(f"\nLoading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Total comments loaded: {len(df):,}")

    df_en = filter_english(df)

    print("\n  Language breakdown before filtering (top 10):")
    print(df_en["lang"].value_counts().head(10).to_string())

    print("\n  Per-tier language retention:")
    for tier, grp in df.groupby("tier"):
        kept = len(df_en[df_en["tier"] == tier])
        pct  = kept / len(grp) * 100
        print(f"    {tier:25s}: {kept:,} / {len(grp):,} kept ({pct:.1f}%)")

    df_balanced = downsample(df_en, TIER_TARGETS, seed=RANDOM_SEED)
    summarise(df_balanced)
    df_balanced.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Columns: {list(df_balanced.columns)}")
