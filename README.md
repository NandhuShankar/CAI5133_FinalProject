# YouTube Audience Emotion Analysis
### Extracting What Tech Audiences Value Using Stratified NLP

**Authors:** Venkata Dinavahi · Michael Montemarano · Nandhu Shankar · Tejaswini Varampati  
**Course:** Social Media Mining — University of South Florida  

---

## Overview

This project investigates whether the emotional and topical patterns in YouTube comments can reveal what different technology audiences value. We collected 171,333 comments across three stratified audience tiers and applied a three-layer NLP pipeline to produce a statistically validated aspect-emotion value map per tier.

---

## The Three-Tier Design

The core methodological contribution is a stratified audience design that isolates audience depth effects rather than treating all YouTube commenters as one population.

| Tier | Label | Description | Example Channels |
|------|-------|-------------|-----------------|
| **Tier 0** | Entertainment Baseline | Non-tech content. Serves as behavioral control group. | Jimmy Fallon, Late Night TV |
| **Tier 1** | General Tech | Broad tech coverage, large non-specialist audiences. | Fireship, MKBHD, Linus Tech Tips |
| **Tier 2** | Niche Tech | Deep specialist communities, single-tool or domain focus. | Vim channels, Linux channels, Security/Hacking |

Tier 0 is used only as an emotion baseline. BERTopic is not run on it since its topics are entertainment-related and irrelevant to the research question.

---

## Key Results

### Emotion Baseline Contrast
All tier comparisons are statistically significant (all p < 0.001 after Bonferroni correction):

| Emotion | Tier 0 (Entertainment) | Tier 1 (General Tech) | Tier 2 (Niche Tech) |
|---------|----------------------|----------------------|---------------------|
| Amusement/Joy | **13.9%** ▲ | 6.9% ▼ | 8.7% ▼ |
| Curiosity/Interest | 6.4% ▼ | **9.5%** ▲ | 7.6% |
| Negative/Critical | 5.5% ▼ | **7.8%** ▲ | 6.3% |
| Admiration/Approval | 13.1% | 12.9% | **14.2%** |
| Neutral/Analytical | 58.2% | 60.1% | **61.0%** |

▲/▼ = significantly higher/lower than expected (standardised residual |z| > 2.0)

Overall: χ²(10) = 449.27, p = 3.00×10⁻⁹⁰, Cramér's V = 0.078

### Comment Engagement Depth
Tech audiences write significantly more per comment than the entertainment baseline:

| Tier | Avg Words | Median Words | Emoji Usage |
|------|-----------|--------------|-------------|
| Tier 0 Baseline | 10.9 | 8 | 19% |
| Tier 1 General Tech | 30.1 | 15 | 9% |
| Tier 2 Niche Tech | 24.0 | 14 | 5% |

Word count differences: Kruskal-Wallis H = 3701.82, p < 0.001  
Emoji usage: χ²(2) = 1380.10, p < 0.001, V = 0.194

### Value Map Highlights (Top Tier 1 Topics)
| Topic | Dominant Non-Neutral Emotion | % |
|-------|------------------------------|---|
| AI / Claude / Anthropic | Curiosity/Interest | 10.6% |
| Vibe Coding / Gemini | Admiration/Approval | 19.2% |
| Creator Content | Admiration/Approval | 18.6% |

### Value Map Highlights (Top Tier 2 Topics)
| Topic | Dominant Non-Neutral Emotion | % |
|-------|------------------------------|---|
| Vim / Nvim / Config | Curiosity/Interest | 13.5% |
| Hacker / Security | Amusement/Joy | 9.6% |
| Code / Programming | Admiration/Approval | 13.3% |

---

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scipy matplotlib transformers torch \
            bertopic sentence-transformers scikit-learn hdbscan langdetect
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Run Pipeline in Order
```bash
# Step 1 — Preprocessing
python scripts/preprocess_comments.py

# Step 2 — Balancing + English filter
python scripts/balance_comments.py

# Step 3 — Three-layer analysis (GPU recommended)
python scripts/analyze_comments.py

# Step 4 — Statistical validation
python scripts/statistics.py

# Step 5 — Visualizations
python scripts/visualize.py
```

### 4. Output Files
| File | Description |
|------|-------------|
| `comments_unified.csv` | All tiers merged, cleaned |
| `comments_balanced.csv` | English-only, downsampled |
| `emotions_tagged.csv` | Per-comment emotion labels |
| `topics_per_tier.csv` | BERTopic topic summaries |
| `aspect_emotion_matrix.csv` | Main result: value map |
| `emotion_baseline_contrast.csv` | Tier emotion distributions |
| `statistics_report.txt` | All statistical tests |
| `charts/` | Visual charts |

---

## Models Used

| Model | Task | Parameters |
|-------|------|-----------|
| [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | Emotion classification (28 labels → 6 macro-groups) | 125M |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence embeddings for BERTopic | 22M |
| BERTopic + HDBSCAN | Unsupervised topic extraction per tier | — |

GoEmotions RoBERTa benchmark: macro-F1 = 0.46 on 28-class test set (human inter-annotator agreement = 0.54).

Claude assisted in creating README.

