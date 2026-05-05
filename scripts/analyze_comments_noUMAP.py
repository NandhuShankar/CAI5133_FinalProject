#Nandhu Shankar U38335248

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

#resolve import errors
def _install(pkg):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
try:
    import torch
except ImportError:
    _install("torch"); import torch
try:
    from transformers import pipeline
except ImportError:
    _install("transformers"); from transformers import pipeline
try:
    from bertopic import BERTopic
except ImportError:
    _install("bertopic"); from bertopic import BERTopic
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _install("sentence-transformers"); from sentence_transformers import SentenceTransformer
try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    _install("scikit-learn"); from sklearn.feature_extraction.text import CountVectorizer
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN


INPUT_CSV        = "comments_balanced.csv"
EMOTION_CSV      = "emotions_tagged.csv"
TOPICS_CSV       = "topics_per_tier.csv"
MATRIX_CSV       = "aspect_emotion_matrix.csv"

MACRO_GROUPS = {
    "admiration/approval":  ["admiration", "approval"],
    "curiosity/interest":   ["curiosity", "desire"],
    "negative/critical":    ["annoyance", "disapproval", "anger", "disgust"],
    "amusement/joy":        ["amusement", "joy", "excitement"],
    "neutral/analytical":   ["neutral", "realization", "surprise"],
    "concern/disappointment": ["disappointment", "sadness", "fear", "grief"],
}

# BERTopic topics per tier
N_TOPICS        = 10      # number of topics to extract per tier
MIN_TOPIC_SIZE  = 10      # minimum comments per topic

BATCH_SIZE      = 32

DEVICE = 0 if torch.cuda.is_available() else -1

TIERS = ["tier0_baseline", "tier1_general_tech", "tier2_niche_tech"]

# Extended stopwords sklearn's default English list and common filler words
CUSTOM_STOPWORDS = list({
    *__import__("sklearn.feature_extraction.text", fromlist=["ENGLISH_STOP_WORDS"])
     .ENGLISH_STOP_WORDS,
    "just", "like", "know", "use", "get", "got", "think", "really", "actually",
    "yeah", "yes", "lol", "haha", "lmao", "omg", "ok", "okay", "great", "good",
    "bad", "thing", "things", "way", "make", "made", "want", "need", "going",
    "come", "came", "said", "say", "look", "looks", "see", "watch", "video",
    "guy", "guys", "people", "man", "time", "work", "works", "working",
    "right", "wrong", "true", "false", "new", "old", "big", "little",
    "literally", "basically", "honestly", "probably", "definitely", "totally",
    "pretty", "kind", "sort", "bit", "lot", "let", "try", "tried", "used",
    "better", "best", "worse", "worst", "nice", "cool", "awesome", "amazing",
    "love", "hate", "feel", "felt", "mean", "means", "sense", "point",
    "im", "ive", "dont", "doesnt", "didnt", "wont", "cant", "isnt", "wasnt",
    "id", "ve", "ll", "re", "wes", "lex",  
})

# The neutral label name — excluded when computing non-neutral emotion ranking
NEUTRAL_LABEL = "neutral/analytical"

def run_emotion_classification(df: pd.DataFrame) -> pd.DataFrame:
    #tag each comment with top emotion, score, and macro emotion
    print("\nEmotion Classification")

    classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=1,
        device=DEVICE,
        truncation=True,
        max_length=128,
    )

    texts = df["text_clean"].tolist()
    results = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Running inference on {len(texts):,} comments "
          f"({total_batches} batches of {BATCH_SIZE})...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_results = classifier(batch)
        results.extend(batch_results)

        if (i // BATCH_SIZE + 1) % 50 == 0:
            pct = (i + BATCH_SIZE) / len(texts) * 100
            print(f"    {min(i + BATCH_SIZE, len(texts)):,} / {len(texts):,} "
                  f"({pct:.0f}%) done")

    top_emotions = [r[0]["label"]  for r in results]
    top_scores   = [r[0]["score"]  for r in results]

    df = df.copy()
    df["top_emotion"] = top_emotions
    df["top_score"]   = top_scores

    label_to_macro = {}
    for macro, labels in MACRO_GROUPS.items():
        for lbl in labels:
            label_to_macro[lbl] = macro

    df["macro_emotion"] = df["top_emotion"].map(
        lambda x: label_to_macro.get(x, "neutral/analytical")
    )

    # per-tier emotion distribution
    print("\n  Emotion distribution per tier (macro):")
    pivot = (df.groupby(["tier", "macro_emotion"])
               .size()
               .unstack(fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).round(3) * 100
    print(pivot_pct.to_string())

    return df

def run_bertopic_per_tier(df: pd.DataFrame) -> pd.DataFrame:

    #Fits a separate BERTopic model per tier.
    print("\n── Layer 2: Topic Extraction (BERTopic per tier)")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    df = df.copy()
    df["topic_id"]    = -1
    df["topic_label"] = "outlier"

    all_topic_rows = []
    # topic_id remains -1 and topic_label remains outlier for Tier 0.
    TOPIC_TIERS = [t for t in TIERS if t != "tier0_baseline"]
    print("Skipping tier0_baseline )")

    for tier in TOPIC_TIERS:
        mask  = df["tier"] == tier
        texts = df.loc[mask, "text_clean"].tolist()
        print(f"\n  [{tier}] Fitting BERTopic on {len(texts):,} comments...")

        vectorizer = CountVectorizer(
            stop_words=CUSTOM_STOPWORDS,
            ngram_range=(1, 2),
            min_df=5,
        )

        # disable UMAP dimensionality reduction before clustering.
        no_umap   = BaseDimensionalityReduction()
        hdbscan   = HDBSCAN(
            min_cluster_size=MIN_TOPIC_SIZE,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=no_umap,      
            hdbscan_model=hdbscan,
            vectorizer_model=vectorizer,
            nr_topics=N_TOPICS,
            min_topic_size=MIN_TOPIC_SIZE,
            verbose=False,
        )

        # L2-normalize embeddings
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)

        topics, _ = topic_model.fit_transform(texts, embeddings)

        idx = df.index[mask]
        df.loc[idx, "topic_id"] = topics

        topic_info = topic_model.get_topic_info()
        id_to_label = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                id_to_label[-1] = "outlier"
            else:
                # Top 3 keywords as label
                keywords = topic_model.get_topic(tid)
                label = " / ".join([kw for kw, _ in keywords[:3]])
                id_to_label[tid] = label

        df.loc[idx, "topic_label"] = df.loc[idx, "topic_id"].map(id_to_label)

        topic_info["tier"] = tier
        topic_info["top_keywords"] = topic_info["Topic"].apply(
            lambda t: ", ".join([kw for kw, _ in topic_model.get_topic(t)[:5]])
            if t != -1 else ""
        )
        all_topic_rows.append(topic_info[["tier", "Topic", "Count",
                                           "Name", "top_keywords"]])

        print(f"    Found {len(topic_info) - 1} topics "
              f"({topic_info[topic_info['Topic']==-1]['Count'].values[0]:,} outliers)")
        print(f"    Top topics:")
        for _, row in topic_info[topic_info["Topic"] != -1].head(5).iterrows():
            print(f"      Topic {row['Topic']:2d} ({row['Count']:4d} comments): "
                  f"{row['top_keywords']}")
    topics_df = pd.concat(all_topic_rows, ignore_index=True)
    topics_df.to_csv(TOPICS_CSV, index=False)
    print(f"\n  Saved topic info → {TOPICS_CSV}")

    return df



def build_aspect_emotion_matrix(df: pd.DataFrame) -> pd.DataFrame:
   #outputs baseline and value map

    print("\n  A) Emotion baseline contrast (all tiers):")
    baseline = (df.groupby(["tier", "macro_emotion"])
                  .size()
                  .unstack(fill_value=0))
    baseline_pct = (baseline.div(baseline.sum(axis=1), axis=0) * 100).round(1)
    print(baseline_pct.to_string())
    baseline_pct.to_csv("emotion_baseline_contrast.csv")
    print("  Saved → emotion_baseline_contrast.csv")

    # value map
    print("\n  B) Aspect-Emotion Value Map (tech tiers only):")
    TOPIC_TIERS = [t for t in TIERS if t != "tier0_baseline"]
    df_clean = df[(df["topic_id"] != -1) & (df["tier"].isin(TOPIC_TIERS))].copy()

    rows = []
    for tier in TOPIC_TIERS:
        t_df = df_clean[df_clean["tier"] == tier]

        for topic_label, grp in t_df.groupby("topic_label"):
            total = len(grp)
            emotion_counts = grp["macro_emotion"].value_counts()
            emotion_pcts   = (emotion_counts / total * 100).round(1)

            # Non-neutral dominant: most common emotion excluding neutral/analytical
            non_neutral = emotion_pcts.drop(NEUTRAL_LABEL, errors="ignore")
            if len(non_neutral) > 0:
                non_neutral_dominant     = non_neutral.idxmax()
                non_neutral_dominant_pct = non_neutral.max()
            else:
                non_neutral_dominant     = "neutral/analytical"
                non_neutral_dominant_pct = emotion_pcts.max()

            row = {
                "tier":                    tier,
                "topic":                   topic_label,
                "n_comments":              total,
                "dominant_emotion":        emotion_pcts.idxmax(),
                "dominant_pct":            emotion_pcts.max(),
                "non_neutral_dominant":    non_neutral_dominant,
                "non_neutral_dominant_pct": non_neutral_dominant_pct,
            }
            for macro in MACRO_GROUPS:
                row[macro] = emotion_pcts.get(macro, 0.0)
            rows.append(row)

    matrix_df = pd.DataFrame(rows).sort_values(
        ["tier", "n_comments"], ascending=[True, False]
    )
    matrix_df.to_csv(MATRIX_CSV, index=False)

    #show both neutral% and non-neutral dominant
    print(f"\n  {'Topic':<32} {'N':>6}  {'Non-Neutral Dominant':<26} {'%':>5}  {'Neutral%':>8}")
    for tier in TOPIC_TIERS:
        print(f"\n  [{tier}]")
        print("  " + "─" * 84)
        for _, row in matrix_df[matrix_df["tier"] == tier].head(8).iterrows():
            print(f"  {row['topic']:<32} {row['n_comments']:>6}  "
                  f"{row['non_neutral_dominant']:<26} {row['non_neutral_dominant_pct']:>5.1f}%"
                  f"  {row[NEUTRAL_LABEL]:>7.1f}%")

    print(f"\n  Saved value map → {MATRIX_CSV}")
    return matrix_df

if __name__ == "__main__":
    if not Path(INPUT_CSV).exists():
        print(f"[ERROR] {INPUT_CSV} not found. Run balance_comments.py first.")
        exit(1)

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df):,} comments across {df['tier'].nunique()} tiers")

    # Layer 1 — skip if already done, reload from saved file
    if Path(EMOTION_CSV).exists():
        print(f"\nFound {EMOTION_CSV} — skipping Layer 1 (loading saved emotions)...")
        df = pd.read_csv(EMOTION_CSV, low_memory=False)
        print(f"Loaded {len(df):,} emotion-tagged comments")
    else:
        df = run_emotion_classification(df)
        df.to_csv(EMOTION_CSV, index=False)
        print(f"\nSaved emotion-tagged data → {EMOTION_CSV}")

    # Layer 2
    df = run_bertopic_per_tier(df)

    # Layer 3
    matrix = build_aspect_emotion_matrix(df)

    # Final combined save
    df.to_csv("comments_full_analysis.csv", index=False)
    print(f"\nFull analysis saved → comments_full_analysis.csv")
    print("\nAll done. Key output files:")
    print(f"  {EMOTION_CSV:<35} — per-comment emotion labels")
    print(f"  {TOPICS_CSV:<35} — BERTopic summary per tier")
    print(f"  {MATRIX_CSV:<35} — aspect-emotion value map (paper result)")
    print(f"  {'comments_full_analysis.csv':<35} — everything combined")
