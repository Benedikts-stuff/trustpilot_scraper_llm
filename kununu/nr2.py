import ast
import pandas as pd
from collections import Counter, defaultdict
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import time
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from typing import List
from prompt import build_classification_prompt, extract_pros_cons_from_reviews, parse_llama_output
from word_list_sentiment import sentiment_score

# === 1. Daten laden ===
file_path = "./kununu_Herma_comments.xlsx"
df_raw = pd.read_excel(file_path)

# Kategorien-Parsing
all_categories = df_raw['Categories'].dropna().apply(ast.literal_eval)
category_with_text_counter = Counter()
category_texts = defaultdict(list)

for entry in all_categories:
    for category, details in entry.items():
        text = details.get("text", "").strip()
        if text:
            category_with_text_counter[category] += 1
            category_texts[category].append(text)

# Top-Kategorien auswählen (hier Platz 2 und 3 der Häufigkeit)
most_common_with_text = category_with_text_counter.most_common()[0:5]
top_categories = [category for category, _ in most_common_with_text]

# DataFrame direkt erstellen
export_df = pd.DataFrame(
    [
        {"Kategorie": category, "Text": text}
        for category in top_categories
        for text in category_texts[category]
    ]
)

# === 2. BERT-Sentiment hinzufügen ===
sentiment_model = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

def bert_sentiment(text):
    res = sentiment_model(text[:512])[0]
    return pd.Series([res["label"], res["score"]])

export_df[["category_BERT", "Score_BERT"]] = export_df["Text"].apply(bert_sentiment)

# === 3. LLaMA-Sentiment hinzufügen ===
def chunk_reviews(reviews: List[str], size: int = 5) -> List[List[str]]:
    return [reviews[i:i+size] for i in range(0, len(reviews), size)]

llama_results = []
review_chunks = chunk_reviews(export_df["Text"].tolist(), size=1)
count =0
fail_count = 0
for chunk in export_df["Text"]:
    review_block = chunk
    print(review_block)
    prompt = build_classification_prompt(review_block)
    if fail_count > 5:
        print("Zu viele Fehler, breche ab.")
        break
    success = False
    while not success:
        try:
            result = extract_pros_cons_from_reviews(prompt)
            parsed = parse_llama_output(result, 1)
            print("parsed: ", parsed)
            llama_results.extend(parsed)
            success = True  # wenn bis hierhin kein Fehler kam
        except Exception as e:
            fail_count += 1
            print(f"Fehler aufgetreten: {e}. Warte 60 Sekunden und versuche erneut...")
            time.sleep(60)

    count += 1
    if count % 20 == 0:
        print(f"Warte nach {count} Anfragen...")
        time.sleep(120)
    time.sleep(5)

# Falls LLaMA-Ergebnisse kürzer sind als DataFrame, auffüllen
while len(llama_results) < len(export_df):
    llama_results.append(None)

export_df["category_llama"] = llama_results

# === 4. Wordlist-Sentiment (SentiWS) hinzufügen ===
def get_wordlist_sentiment(text):
    score = sentiment_score(text)
    if score > 0:
        return "positive", score
    elif score < 0:
        return "negative", score
    else:
        return "neutral", score

export_df[["category_wordlist", "score_wordlist"]] = export_df["Text"].apply(
    lambda t: pd.Series(get_wordlist_sentiment(t))
)

# === 5. In Excel speichern ===
export_df.to_excel(
    "sentiment_combined_results_Herma.xlsx",
    index=False,
    columns=[
        "Kategorie", "Text",
        "category_BERT", "Score_BERT",
        "category_llama",
        "category_wordlist", "score_wordlist"
    ]
)

print("Export abgeschlossen: sentiment_combined_results_Herma.xlsx")
