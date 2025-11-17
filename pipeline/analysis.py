# analysis.py
import re
import time
import pandas as pd
from collections import Counter, defaultdict
from typing import List

# --- Nötige Bibliotheken für die Analysen ---
from transformers import pipeline
from groq import Groq
import spacy
import nltk
from nltk.corpus import stopwords
import streamlit as st  # <-- WICHTIGER NEUER IMPORT

# Importiere unsere zentrale Konfiguration
# (Stelle sicher, dass config.py noch existiert, ODER
# verschiebe die API-Keys und Pfade hierher)
import config


# --- Modell-Ladefunktionen mit CACHING ---
# Dies ist die neue, schnelle Methode.
# Die Modelle werden erst geladen, wenn sie gebraucht werden,
# und bleiben dann im Speicher.

@st.cache_resource
def get_bert_model():
    """Lädt und cached das BERT-Modell."""
    print("Lade BERT-Modell (dies passiert nur einmal)...")
    try:
        return pipeline(
            "sentiment-analysis",
            model="oliverguhr/german-sentiment-bert"
        )
    except Exception as e:
        st.error(f"Fehler beim Laden des BERT-Modells: {e}")
        return None


@st.cache_resource
def get_spacy_model():
    """Lädt und cached das Spacy-Modell."""
    print("Lade Spacy-Modell (dies passiert nur einmal)...")
    try:
        return spacy.load("de_core_news_sm")
    except Exception as e:
        st.error(f"Fehler beim Laden des Spacy-Modells: {e}")
        st.info("Bitte 'python -m spacy download de_core_news_sm' im Terminal ausführen.")
        return None


@st.cache_resource
def get_senti_dict():
    """Lädt und cached die SentiWS-Wortlisten."""
    print("Lade SentiWS-Wortlisten (dies passiert nur einmal)...")
    senti_dict = {}

    def load_sentiws(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    lemma_pos = parts[0]
                    score = float(parts[1])
                    lemma = lemma_pos.split("|")[0].lower()
                    senti_dict[lemma] = score
                    if len(parts) == 3:
                        for form in parts[2].split(","):
                            senti_dict[form.lower()] = score

    try:
        load_sentiws(config.SENTIWS_POS_PATH)
        load_sentiws(config.SENTIWS_NEG_PATH)
        nltk.download('stopwords')
        return senti_dict, set(stopwords.words('german'))
    except FileNotFoundError as e:
        st.error(f"Fehler: SentiWS-Datei nicht gefunden: {e}")
        return {}, set()


@st.cache_resource
def get_llama_client():
    """Initialisiert und cached den Groq-Client."""
    print("Initialisiere LLaMA (Groq) Client...")
    return Groq(api_key=config.GROQ_API_KEY)


# =======================================================================
#  1. ANALYSE-FUNKTIONEN: WORTLISTE (SentiWS)
# =======================================================================

def tokenize_and_filter(text):
    _, STOP_WORDS_DE = get_senti_dict()
    if not isinstance(text, str): return []
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS_DE]


def calculate_sentiment_score(text: str) -> dict:
    senti_dict, _ = get_senti_dict()
    if not senti_dict: return {"label": "error", "score": 0}

    tokens = tokenize_and_filter(text)
    score = 0
    for token in tokens:
        if token in senti_dict:
            score += senti_dict[token]

    label = "neutral"
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    return {"label": label, "score": score}


def run_wordlist_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    print("Starte Wortlisten-Analyse...")
    results = df[text_column].apply(calculate_sentiment_score)
    df['Wortliste_Sentiment'] = [res['label'] for res in results]
    df['Wortliste_Score'] = [res['score'] for res in results]
    print("Wortlisten-Analyse abgeschlossen.")
    return df


# =======================================================================
#  2. ANALYSE-FUNKTIONEN: BERT (Transformers)
# =======================================================================

def calculate_bert_sentiment(text: str, model) -> dict:
    """Analysiert einen Text mit dem (jetzt übergebenen) BERT-Modell."""
    try:
        result = model(text[:512])[0]
        return result
    except Exception as e:
        print(f"BERT-Analysefehler: {e}")
        return {"label": "error", "score": 0.0}


def run_bert_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    # HOLT DAS MODELL AUS DEM CACHE (oder lädt es beim ersten Mal)
    model = get_bert_model()
    if model is None:
        st.error("BERT-Modell konnte nicht geladen werden. Überspringe...")
        return df

    print("Starte BERT-Analyse...")
    results = df[text_column].apply(lambda text: calculate_bert_sentiment(text, model))
    df['BERT_Sentiment'] = [res['label'] for res in results]
    df['BERT_Score'] = [res['score'] for res in results]
    print("BERT-Analyse abgeschlossen.")
    return df


# =======================================================================
#  3. ANALYSE-FUNKTIONEN: LLAMA (Groq)
# =======================================================================

def build_llama_prompt(review_block: str) -> str:
    # (Keine Änderung hier)
    prompt = f"""Du erhältst eine Bewertung von einem Mitarbeiter.
Ordne diese Bewertung einer der folgenden vier Kategorien zu:
- positiv
- neutral
- negativ
- Verbesserungsvorschlag

Antworte NUR mit dem exakten Label (positiv, neutral, negativ, Verbesserungsvorschlag).
Keine Zusätze, keine Erklärungen, nur das eine Wort.

Hier ist die Bewertung:
{review_block.strip()}
"""
    return prompt


def parse_llama_output(llama_response: str) -> str:
    # (Keine Änderung hier)
    line = llama_response.strip().lower()
    if line.startswith("+"): line = line[1:].strip()
    if line in {"positiv", "neutral", "negativ", "verbesserungsvorschlag"}: return line
    if "positiv" in line: return "positiv"
    if "negativ" in line: return "negativ"
    if "neutral" in line: return "neutral"
    if "verbesserungsvorschlag" in line: return "verbesserungsvorschlag"
    return "unbekannt"


def classify_with_llama(text: str, client) -> str:
    """Führt eine einzelne Klassifizierung mit LLaMA durch."""
    prompt = build_llama_prompt(text)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "Du bist ein HR Analyst. Antworte IMMER nur mit dem einen Wort der Kategorie."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        response_text = chat_completion.choices[0].message.content
        return parse_llama_output(response_text)

    except Exception as e:
        print(f"LLaMA-Analysefehler: {e}")
        if "rate limit" in str(e).lower():
            print("Rate-Limit erreicht. Warte 60 Sekunden...")
            time.sleep(60)
            return classify_with_llama(text, client)  # Erneuter Versuch
        return "error"


def run_llama_classification(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    # HOLT DEN CLIENT AUS DEM CACHE
    client = get_llama_client()
    if client is None:
        st.error("Groq-Client konnte nicht initialisiert werden. Überspringe...")
        return df

    print("Starte LLaMA-Analyse...")
    results = []
    total = len(df)
    progress_bar = st.progress(0, text="LLaMA-Analyse läuft...")

    for i, row in df.iterrows():
        text = row[text_column]
        print(f"Bearbeite LLaMA {i + 1}/{total}...")
        label = classify_with_llama(text, client)
        results.append(label)

        # Update Progress Bar
        progress_bar.progress((i + 1) / total, text=f"LLaMA-Analyse: {i + 1}/{total}")

        time.sleep(1)  # Rate-Limit
        if (i + 1) % 20 == 0:
            print("Warte 30 Sekunden nach 20 Anfragen...")
            time.sleep(30)

    progress_bar.empty()  # Entfernt die Bar nach Abschluss
    df['LLaMA_Kategorie'] = results
    print("LLaMA-Analyse abgeschlossen.")
    return df


# =======================================================================
#  4. ANALYSE-FUNKTIONEN: ASPEKT-ANALYSE
# =======================================================================

def lemmatize_text(text: str, nlp_model) -> str:
    """Lemmatisiert einen Text mit dem (übergebenen) Spacy-Modell."""
    if not isinstance(text, str): return ""

    doc = nlp_model(text.lower())
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(lemmas)


def count_aspect_sentiment(lemmatized_text: str, aspect_words: dict) -> dict:
    # (Keine Änderung hier)
    score = {"pos": 0, "neg": 0}
    for label in ["pos", "neg"]:
        for word in aspect_words[label]:
            if word in lemmatized_text:
                score[label] += 1
    return score


def run_aspect_analysis(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    # HOLT DAS MODELL AUS DEM CACHE
    nlp = get_spacy_model()
    if nlp is None:
        st.error("Spacy-Modell konnte nicht geladen werden. Überspringe...")
        return df

    print("Starte Aspekt-Analyse...")
    print("Schritt 1: Lemmatisierung...")
    df['lemmatized'] = df[text_column].apply(lambda text: lemmatize_text(text, nlp))

    print("Schritt 2: Aspekte zählen...")
    for aspect, words in config.ASPECTS.items():
        pos_col = f"{aspect}_pos"
        neg_col = f"{aspect}_neg"

        scores = df['lemmatized'].apply(lambda text: count_aspect_sentiment(text, words))

        df[pos_col] = [s['pos'] for s in scores]
        df[neg_col] = [s['neg'] for s in scores]

    print("Aspekt-Analyse abgeschlossen.")
    return df