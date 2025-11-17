from scraper import scrape_trustpilot_reviews
from groq import Groq
import pandas as pd
import time
from collections import defaultdict
from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer, util


# Groq Setup
api_key = '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'
client = Groq(api_key=api_key)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_classification_prompt(review_block: str) -> str:
    prompt = f"""Du erhältst eine Bewertung von einem Mitarbeiter zu einem Unternehmen. 
Der Bewertungstext steht in einer eigenen Zeile und beginnt mit einem `+`. Die sollte du einer von vie
Kategorien zuordnen.

Kategorien (nur diese Wörter beziehungsweise <Label> sind erlaubt):
- positiv
- neutral
- negativ
- Verbesserungsvorschlag

Gib die Antwort **nur** in folgendem Format zurück:
Klar gerne. Hier sind die klassifizierten Bewertungen:
#
+ <Label>
#
...

<Label> muss exakt einer der vier Kategorien sein. 
Es dürfen keine anderen Wörter, Texte, Erklärungen oder Kommentare vorkommen.
Gib auf keinen Fall den Bewertungstext oder einzelne Buchstaben zurück.
GEBE AUF KEINEN FALL MEHR KATEGORIEN ALS DIR BEWERTUNGEN VORLIEGEN das wäre Katastrophal!
Hier ist die Bewertungsliste:

{review_block.strip()}
"""
    return prompt




def extract_pros_cons_from_reviews(review) -> str:

    prompt = build_classification_prompt(review)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein HR Analyst für Mitarbeiterfeedback. "
                    "Deine Aufgabe ist es, Mitarbeiterrezensionen zu analysieren und zu klassifizieren, "
                    "ob sie Positiv, Neutral, negativ oder ein Verbesserungsvorschlag sind. "
                    "Achte auf die Struktur der Ausgabe und halte dich genau an das vorgegebene Format. "
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


def parse_llama_output(llama_response: str, size) -> list:
    """
    Extrahiert alle + Sentiment-Zeilen aus dem LLaMA-Output.
    Ignoriert überzählige # und leere Zeilen.
    """
    count = 0
    lines = llama_response.strip().splitlines()
    results = []

    for line in lines:
        line = line.strip()
        if line.startswith("+") and count<size:
            sentiment = line[1:].strip().lower()
            if sentiment in {"Verbesserungsvorschlag", "positiv", "neutral", "negativ"}:
                results.append(sentiment)
                count+=1

    return results
