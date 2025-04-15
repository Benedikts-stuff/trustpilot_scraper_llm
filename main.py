from scraper import scrape_trustpilot_reviews
from groq import Groq
from groq.resources import embeddings
import os
import pandas as pd
import time
from collections import Counter
import re
from collections import defaultdict
from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer, util

base_url = 'https://de.trustpilot.com/review/www.energieag.at'

reviews = scrape_trustpilot_reviews(base_url)


def export_reviews_to_excel(reviews: list[dict], filename: str = "trustpilot_reviews.xlsx"):
    """
    Exportiert eine Liste von Review-Dictionaries in eine Excel-Datei.

    Args:
        reviews (list of dict): Die Bewertungen im Format [{'Date': ..., 'Author': ..., ...}, ...]
        filename (str): Der Name der Excel-Datei (Standard: trustpilot_reviews.xlsx)
    """
    df = pd.DataFrame(reviews)

    columns_order = ['Date', 'Author', 'Location', 'Rating', 'Heading', 'Body']
    df = df[columns_order]

    df.to_excel(filename, index=False)
    print(f"Export erfolgreich: {filename}")







# Groq Setup
api_key = 'API_KEY'
client = Groq(api_key=api_key)

# Sentence Transformer Modell
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_reviews(reviews: List[dict], batch_size: int = 5):
    for i in range(0, len(reviews), batch_size):
        yield reviews[i:i + batch_size]


def build_classification_prompt(review_text: str, categories_with_defs: dict) -> str:
    cat_block = "\n".join(f"- {cat}: {definition}" for cat, definition in categories_with_defs.items())

    prompt = (
        "Analysiere die folgende Kundenrezensionen (das sind 5 Rezensionen zusammen zur Info) im Hinblick auf die folgenden Qualitätsmerkmale eines Stromanbieters:\n\n"
        f"{cat_block}\n\n"
        "Gib für jede Kategorie an, ob sie in der Rezension\n"
        "positiv erwähnt wurde (+)\n"
        "negativ erwähnt wurde (-)\n"
        "oder nicht erwähnt wurde (/)\n\n"
        f"Rezensionen:\n{review_text}\n\n"
        "Antwortformat (bitte genau so):\n"
        + "\n".join(f"{cat}: +" for cat in categories_with_defs.keys())  # Beispiel-Output
    )
    return prompt

CATEGORIES_WITH_DEFS = {
    "Kundenservice": "Wie freundlich, hilfsbereit und erreichbar der Kundensupport ist.",
    "Preis-Leistungs-Verhältnis": "Ob der Preis in einem angemessenen Verhältnis zur gebotenen Leistung steht.",
    "Zuverlässigkeit der Abrechnung": "Ob Abrechnungen korrekt, pünktlich und nachvollziehbar sind.",
    "Vertragsklarheit": "Wie verständlich, transparent und fair die Vertragsbedingungen sind.",
    "Wechselprozess": "Wie einfach und reibungslos der Wechsel zum Anbieter abläuft.",
    "Kommunikation": "Wie klar, schnell und informativ die Kommunikation mit dem Anbieter ist.",
    "Technischer Support": "Ob technische Probleme kompetent und zeitnah gelöst werden.",
    "Nachhaltigkeit / Ökostrom": "Ob der Anbieter umweltfreundliche Energiequellen nutzt und nachhaltig agiert.",
    "Nichts": "Wenn keine der anderen Kategorien passt."
}

def extract_pros_cons_from_reviews(review) -> str:

    prompt = build_classification_prompt(review, CATEGORIES_WITH_DEFS)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein Analyst für Kundenfeedback. "
                    "Deine Aufgabe ist es, Kundenrezensionen zu analysieren und die wichtigsten "
                    "positiven und negativen Aspekte zusammenzufassen. "
                    "Achte auf Wiederholungen und fasse ähnliche Aussagen zusammen."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-70b-8192"
    )

    return chat_completion.choices[0].message.content


def parse_review_classification(response: str, categories: list[str]) -> dict:
    result = defaultdict(lambda: "/")
    for line in response.strip().splitlines():
        if ":" in line:
            cat, val = line.split(":", 1)
            cat = cat.strip()
            val = val.strip()
            if cat in categories and val in {"+", "-", "/"}:
                result[cat] = val
    return dict(result)


def count_category_results(all_reviews: list[dict], categories: list[str]) -> dict:
    counts = {cat: {"+": 0, "-": 0, "/": 0} for cat in categories}

    for r in all_reviews:
        for cat in categories:
            val = r.get(cat, "/")
            if val in counts[cat]:
                counts[cat][val] += 1

    return counts


def analyze_all_reviews_structured(reviews: List[dict]) -> list[dict]:
    """
    Gibt eine Liste von Dicts zurück mit Klassifikation pro Review.
    """
    count = 0
    parsed_reviews = []
    for review in reviews:
        print("Review Nummer: ",count)
        try:
            result = extract_pros_cons_from_reviews(review)
            parsed = parse_review_classification(result, list(CATEGORIES_WITH_DEFS.keys()))
            parsed_reviews.append(parsed)
            time.sleep(0.5)
            count +=1
        except Exception as e:
            print(f"Fehler bei Batch: {e}")
    return parsed_reviews

def export_category_summary_to_excel(counts: dict, filename: str = "category_summary.xlsx"):
    """
    Exportiert die Kategoriezählungen in eine Excel-Datei.
    """
    data = []
    for category, value_counts in counts.items():
        row = {
            "Kategorie": category,
            "Positive_count": value_counts.get("+", 0),
            "Negative_count": value_counts.get("-", 0),
            "Not_affected_count": value_counts.get("/", 0)
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f" Excel-Datei gespeichert unter: {filename}")


parsed_reviews = analyze_all_reviews_structured(reviews)
category_counts = count_category_results(parsed_reviews, list(CATEGORIES_WITH_DEFS.keys()))
export_category_summary_to_excel(category_counts)

