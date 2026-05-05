
#Nandhu Shankar
#Since we had comment files from different types this file was used to merge them inot a single data souce.

import json
import re
import pandas as pd
from pathlib import Path

#for the tier1 text comment
def load_txt(path: str, tier_label: str = "tier0_baseline") -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append({
                    "comment_id": None,
                    "tier": tier_label,
                    "channel_source": Path(path).stem,
                    "video_id": None,
                    "video_title": None,
                    "author": None,
                    "text": text,
                    "likes": None,
                    "date": None,
                })
    return pd.DataFrame(rows)

#for json and jsonl
def load_json(path: str, tier_label: str = "tier1_general_tech") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["tier"] = tier_label
    df = _normalise_columns(df)
    return df


def load_jsonl(path: str, tier_label: str = "tier2_niche_tech") -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["tier"] = tier_label
    df = _normalise_columns(df)
    return df

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = ["comment_id", "tier", "channel_source", "video_id",
                 "video_title", "author", "text", "likes", "date"]
    for col in canonical:
        if col not in df.columns:
            df[col] = None
    return df[canonical]


#clean text of white spance and weird characters.
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text_clean"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} empty rows after cleaning.")

    # Drop exact duplicates within the same tier
    before = len(df)
    df = df.drop_duplicates(subset=["tier", "text_clean"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} exact-duplicate comments.")

    # Derived features
    df["char_count"]  = df["text_clean"].str.len()
    df["word_count"]  = df["text_clean"].str.split().str.len()
    df["has_emoji"]   = df["text_clean"].str.contains(
        r"[\U0001F300-\U0001FAFF]", regex=True, na=False)
    df["is_multilingual"] = df["text_clean"].str.contains(
        r"[^\x00-\x7F]", regex=True, na=False)

    return df

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("tier").agg(
        total_comments      = ("text_clean", "count"),
        avg_word_count      = ("word_count", "mean"),
        median_word_count   = ("word_count", "median"),
        avg_char_count      = ("char_count", "mean"),
        pct_has_emoji       = ("has_emoji", "mean"),
        pct_multilingual    = ("is_multilingual", "mean"),
        unique_authors      = ("author", "nunique"),
    ).round(2).reset_index()

    summary["pct_has_emoji"]    = (summary["pct_has_emoji"]    * 100).round(1)
    summary["pct_multilingual"] = (summary["pct_multilingual"] * 100).round(1)
    return summary

if __name__ == "__main__":

    TXT_FILE   = "tier1comments.txt"         
    JSON_FILE  = "tier2comments.json"          
    JSONL_FILE = "tier3comments.jsonl"   
    OUTPUT_CSV = "comments_unified.csv"

    dfs = []

    if Path(TXT_FILE).exists():
        print(f"\nLoading Tier 0 baseline: {TXT_FILE}")
        dfs.append(load_txt(TXT_FILE, tier_label="tier0_baseline"))
    else:
        print(f"[WARN] {TXT_FILE} not found — skipping.")

    if Path(JSON_FILE).exists():
        print(f"Loading Tier 1 general tech: {JSON_FILE}")
        dfs.append(load_json(JSON_FILE, tier_label="tier1_general_tech"))
    else:
        print(f"[WARN] {JSON_FILE} not found — skipping.")

    if Path(JSONL_FILE).exists():
        print(f"Loading Tier 2 niche tech: {JSONL_FILE}")
        dfs.append(load_jsonl(JSONL_FILE, tier_label="tier2_niche_tech"))
    else:
        print(f"[WARN] {JSONL_FILE} not found — skipping.")

    if not dfs:
        print("No files loaded. Check your paths above.")
        exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nRaw total comments loaded: {len(combined)}")

    print("\nCleaning...")
    combined = preprocess(combined)
    print(f"Final comment count after cleaning: {len(combined)}")

    print("\n── Per-Tier Summary ")
    summary = summarise(combined)
    print(summary.to_string(index=False))

    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved unified dataset → {OUTPUT_CSV}")
    print(f"Columns: {list(combined.columns)}")
