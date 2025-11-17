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
from prompt import build_classification_prompt, extract_pros_cons_from_reviews,parse_llama_output
from word_list_sentiment import sentiment_score

# Excel-Datei laden
file_path = "./kununu_phoenix_contact_comments.xlsx"
df = pd.read_excel(file_path)

# Kategorien als echte Dictionaries parsen
all_categories = df['Categories'].dropna().apply(ast.literal_eval)

# Zähler und Textsammlung vorbereiten
category_with_text_counter = Counter()
category_texts = defaultdict(list)

# Kategorien durchgehen
for entry in all_categories:
    for category, details in entry.items():
        text = details.get("text", "").strip()
        if text:
            category_with_text_counter[category] += 1
            category_texts[category].append(text)

# Top 5 Kategorien mit Text
most_common_with_text = category_with_text_counter.most_common()[1:3]
top_categories = [category for category, _ in most_common_with_text]

# Daten für Export vorbereiten
export_data = []
for category in top_categories:
    for text in category_texts[category]:
        export_data.append({
            "Kategorie": category,
            "Text": text
        })

# Sentiment-Analyse Pipeline laden
sentiment_model = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
for entry in export_data:
    result = sentiment_model(entry["Text"][:512])[0]
    entry["Sentiment"] = result["label"]
    entry["Score"] = result["score"]

# Als DataFrame formatieren
export_df = pd.DataFrame(export_data)

def chunk_reviews(reviews: List[str], size: int = 5) -> List[List[str]]:
    return [reviews[i:i+size] for i in range(0, len(reviews), size)]

# Extrahiere nur die Texte
all_reviews = [
    f"Kategorie: {row['Kategorie']}, Bewertungstext: {row['Text']}"
    for _, row in export_df.iterrows()
]
df_reviews = pd.DataFrame(all_reviews, columns=["Text"])
df_reviews.to_excel("all_reviews.xlsx", index=False)

for i, row in export_df.iterrows():
    print(i, row["Kategorie"], row["Text"][:30])

print(len(all_reviews))
review_chunks = chunk_reviews(all_reviews, size=5)
print(len(review_chunks))

llama_ratings = []
for chunk in review_chunks:
    break
    review_block = "\n".join(f"+ {text.strip()}" for text in chunk)
    prompt = build_classification_prompt(review_block)
    result = extract_pros_cons_from_reviews(prompt)
    print(result)
    parsed=parse_llama_output(result)
    print(parsed)
    llama_ratings.extend(parsed)
    time.sleep(2)
print(llama_ratings)
print("llama ratings len", len(llama_ratings))
ratings_df = pd.DataFrame(llama_ratings, columns=["Sentiment"])
ratings_df.to_excel("llama_ratings.xlsx", index=False)

wordlist_ratings = []
for review in df_reviews["Text"]:
    score = sentiment_score(review)
    if score > 0:
        wordlist_ratings.append((review, "positive", score))
    elif score < 0:
        wordlist_ratings.append((review, "negative", score))
    else:
        wordlist_ratings.append((review, "neutral", score))


df = pd.DataFrame(wordlist_ratings, columns=["review", "category", "score"])
df.to_excel("sentiment_wordlist_results.xlsx", index=False)

#export_df['LLama_Sentiment'] = llama_ratings
# In Excel speichern
export_df.to_excel("top5_kategorien_mit_texten_llama.xlsx", index=False)

print("Export abgeschlossen: top5_kategorien_mit_texten.xlsx")








sentiment_counts = export_df.groupby(["Kategorie", "Sentiment"]).size().unstack().fillna(0)
sentiment_counts.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Sentiment-Verteilung pro Kategorie")
plt.ylabel("Anzahl Texte")
plt.xlabel("Kategorie")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Alle negativen Texte zusammenfassen
negative_texts = " ".join(export_df[export_df['Sentiment'] == 'negative']['Text'])

stopwords_de = set(stopwords.words('german'))

# Wörter filtern
filtered_words = " ".join([
    word for word in negative_texts.lower().split()
    if word not in stopwords_de and word.isalpha()
])

# Wordcloud generieren
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)

# Anzeigen
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Häufige Wörter in negativen Bewertungen (ohne Stoppwörter)")
plt.show()
