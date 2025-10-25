import re
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

DATA_FILENAME = "Alice’s Adventures in Wonderland.txt"
ALT_PATH = "/mnt/data/Alice’s Adventures in Wonderland.txt"

OUTPUT_TFIDF_CSV = "tfidf_top20_per_chapter.csv"
OUTPUT_LDA_TOPICS_CSV = "lda_topics.csv"
OUTPUT_CHAPTER_TOPIC_CSV = "chapter_topic_distribution.csv"

CUSTOM_STOPWORDS = {
    "im", "dont", "cant", "couldnt", "wouldnt", "ive", "ill", "youre", "youve", "youd",
    "hes", "shes", "thats", "theres", "whats", "havent", "isnt", "arent", "wasnt", "werent",
    "didnt", "wont", "lets", "one", "two", "say", "said", "would", "could", "must"
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

WORD_RE = re.compile(r"[a-z]+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = (text.replace("’", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))
    text = text.replace("_", " ")
    return text


def tokenize(text: str):
    text = normalize_text(text)
    tokens = WORD_RE.findall(text)
    return [t for t in tokens if (t not in STOPWORDS and len(t) >= 2)]


def analyzer(text: str):
    return tokenize(text)


CHAPTER_LINE_RE = re.compile(
    r"^\s*chapter\s+([ivxlcdm]+)\.?\b[^\n]*$",
    flags=re.IGNORECASE | re.MULTILINE
)


def _token_count(text: str) -> int:
    return len(analyzer(text))


def split_into_chapters(full_text: str, min_tokens: int = 150):
    text = full_text

    contents_re = re.compile(r"^.*?\bcontents\b.*?(?=^\s*chapter\s+i\b)",
                             flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    text = contents_re.sub("", text)

    matches = list(CHAPTER_LINE_RE.finditer(text))
    if not matches:
        return [("FULL_TEXT", text)]

    ends = [m.start() for m in matches[1:]] + [len(text)]
    chapters_raw = []
    for i, (m, end_pos) in enumerate(zip(matches, ends)):
        start_body = m.end()
        title_line = m.group(0).strip()
        body = text[start_body:end_pos].strip()
        tokc = _token_count(body)
        roman = m.group(1).lower()
        chapters_raw.append({
            "i": i,
            "roman": roman,
            "title": re.sub(r"\s+", " ", title_line),
            "body": body,
            "tokc": tokc
        })

    chapters_raw = [c for c in chapters_raw if c["tokc"] >= min_tokens]
    if not chapters_raw:
        return [("FULL_TEXT", text)]

    best_by_roman = {}
    for c in chapters_raw:
        r = c["roman"]
        if (r not in best_by_roman) or (c["tokc"] > best_by_roman[r]["tokc"]):
            best_by_roman[r] = c

    final = sorted(best_by_roman.values(), key=lambda x: x["i"])
    return [(c["title"], c["body"]) for c in final]


def tfidf_top_terms_per_doc(docs: List[str], top_k: int = 20):
    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=2)
    X = vectorizer.fit_transform(docs)
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        idx = np.argsort(row)[::-1][:top_k]
        for rank, j in enumerate(idx, start=1):
            rows.append({"doc_id": i, "rank": rank, "term": terms[j], "tfidf": float(row[j])})
    df = pd.DataFrame(rows)
    return df, vectorizer


def lda_topics(docs: List[str], n_topics: int = 6, n_top_words: int = 12, max_df: float = 0.95, min_df: int = 2):
    cvec = CountVectorizer(analyzer=analyzer, max_df=max_df, min_df=min_df)
    Xc = cvec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='batch', random_state=42, max_iter=30)
    W = lda.fit_transform(Xc)
    H = lda.components_
    vocab = np.array(cvec.get_feature_names_out())

    topic_rows = []
    for k in range(n_topics):
        comp = H[k]
        idx = np.argsort(comp)[::-1][:n_top_words]
        for rank, j in enumerate(idx, start=1):
            topic_rows.append({"topic": k, "rank": rank, "word": vocab[j], "weight": float(comp[j])})
    topics_df = pd.DataFrame(topic_rows)

    doc_topic = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    doc_topic_df = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(n_topics)])
    return topics_df, doc_topic_df


def compare_tfidf_lda(tfidf_df: pd.DataFrame, lda_topics_df: pd.DataFrame, top_k_overlap: int = 20) -> pd.DataFrame:
    tfidf_sets = {
        i: set(tfidf_df[tfidf_df["doc_id"] == i].sort_values("rank").head(top_k_overlap)["term"].tolist())
        for i in tfidf_df["doc_id"].unique()
    }
    topic_sets = {
        k: set(lda_topics_df[lda_topics_df["topic"] == k].sort_values("rank").head(top_k_overlap)["word"].tolist())
        for k in lda_topics_df["topic"].unique()
    }
    rows = []
    for i, terms_i in tfidf_sets.items():
        for k, words_k in topic_sets.items():
            overlap = len(terms_i & words_k)
            rows.append({"doc_id": int(i), "topic": int(k), "overlap": int(overlap)})
    return pd.DataFrame(rows)


def main(data_path: str = None, n_topics: int = 6, top_k_tfidf: int = 20, n_top_words_topic: int = 12):
    if data_path is None:
        if os.path.exists(DATA_FILENAME):
            data_path = DATA_FILENAME
        elif os.path.exists(ALT_PATH):
            data_path = ALT_PATH
        else:
            raise FileNotFoundError("Could not find the text file.")
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    chapters = split_into_chapters(raw_text)
    titles = [t for t, _ in chapters]
    docs = [c for _, c in chapters]

    print(f"Found {len(docs)} chapters:")
    for i, title in enumerate(titles[:len(docs)]):
        print(f"  [{i}] {title}")

    tfidf_df, _ = tfidf_top_terms_per_doc(docs, top_k=top_k_tfidf)
    tfidf_df["chapter_title"] = tfidf_df["doc_id"].apply(lambda i: titles[i])

    lda_topics_df, doc_topic_df = lda_topics(docs, n_topics=n_topics, n_top_words=n_top_words_topic)

    # Save CSVs
    tfidf_df.to_csv(OUTPUT_TFIDF_CSV, index=False)
    lda_topics_df.to_csv(OUTPUT_LDA_TOPICS_CSV, index=False)
    ct = doc_topic_df.copy()
    ct.insert(0, "doc_id", range(len(ct)))
    ct.insert(1, "chapter_title", titles)
    ct.to_csv(OUTPUT_CHAPTER_TOPIC_CSV, index=False)

    # Console summaries
    print("\n=== TF-IDF: Top-5 terms for first 3 chapters ===")
    for i in range(min(3, len(docs))):
        top_terms = (tfidf_df[tfidf_df["doc_id"] == i].sort_values("rank").head(5)["term"].tolist())
        print(f"[{i}] {titles[i]} -> {', '.join(top_terms)}")

    print("\n=== LDA: Top words per topic ===")
    for k in sorted(lda_topics_df["topic"].unique()):
        top_words = (lda_topics_df[lda_topics_df["topic"] == k].sort_values("rank").head(10)["word"].tolist())
        print(f"Topic {k}: {', '.join(top_words)}")

    print("\n=== Chapter -> Dominant topic ===")
    dom = doc_topic_df.values.argmax(axis=1)
    for i, k in enumerate(dom):
        print(f"[{i:02d}] {titles[i]} => topic {k}")

    # Overlap heuristic
    overlap_df = compare_tfidf_lda(tfidf_df, lda_topics_df, top_k_overlap=min(top_k_tfidf, n_top_words_topic))
    print("\n=== For each chapter: TF-IDF vs LDA topic overlap (best) ===")
    for i in sorted(tfidf_df['doc_id'].unique()):
        sub = overlap_df[overlap_df['doc_id'] == i].sort_values('overlap', ascending=False)
        if sub.empty:
            continue
        best_topic = int(sub.iloc[0]['topic'])
        best_overlap = int(sub.iloc[0]['overlap'])
        print(f"[{i:02d}] {titles[i]} -> best_topic={best_topic}, overlap={best_overlap}")

    return {
        "tfidf_csv": os.path.abspath(OUTPUT_TFIDF_CSV),
        "lda_topics_csv": os.path.abspath(OUTPUT_LDA_TOPICS_CSV),
        "chapter_topic_csv": os.path.abspath(OUTPUT_CHAPTER_TOPIC_CSV),
        "num_chapters": len(docs),
        "titles": titles
    }


out = main()
print("\nSaved files:")
for K, V in out.items():
    if K.endswith("_csv"):
        print(f" - {K}: {V}")
