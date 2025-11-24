# reporting.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO

# Setze den Style für alle Plots
sns.set_theme(style="whitegrid")


def create_wordcloud(text_series, title):
    """Erstellt eine Wordcloud aus einer Pandas-Textserie."""
    # (Dein bestehender Code)
    from nltk.corpus import stopwords
    import nltk
    try:
        stop_words = set(stopwords.words('german'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('german'))

    full_text = " ".join(text_series.dropna().astype(str))

    filtered_words = " ".join([
        word for word in full_text.split()
        if word.lower() not in stop_words and len(word) > 2
    ])

    if not filtered_words:
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    return fig


def plot_llama_distribution(df):
    """Zeigt LLaMA-Kategorien an und erzwingt alle 4 Labels."""
    if 'LLaMA_Kategorie' not in df.columns: return None

    # Definierte Reihenfolge erzwingen
    categories = ["positiv", "neutral", "negativ", "Verbesserungsvorschlag"]

    # Zählen und Reindexieren (damit 0-Werte auftauchen)
    counts = df['LLaMA_Kategorie'].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("LLaMA KI-Klassifizierung")
    ax.set_ylabel("Anzahl Reviews")
    ax.set_xlabel("")
    return fig


def plot_sentiment_timeline(df):
    """Zeigt den Verlauf des BERT-Sentiments über die Zeit."""
    if 'Date' not in df.columns or 'BERT_Score' not in df.columns: return None

    # Kopie machen und Datum konvertieren
    df_plot = df.copy()
    df_plot['Date'] = pd.to_datetime(df_plot['Date'], errors='coerce')
    df_plot = df_plot.dropna(subset=['Date']).sort_values('Date')

    if df_plot.empty: return None

    # Wir glätten die Kurve (Rolling Average über 7 Einträge), damit man Trends sieht
    df_plot['Rolling_Score'] = df_plot['BERT_Score'].rolling(window=5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_plot, x='Date', y='Rolling_Score', marker="o", ax=ax, color="tab:blue",
                 label="Sentiment Trend")

    # Referenzlinie bei 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Sentiment-Entwicklung über die Zeit")
    ax.set_ylabel("Sentiment Score (negativ < 0 < positiv)")
    return fig


def plot_star_distribution(df):
    """Zeigt die Verteilung der Sterne-Bewertungen (falls vorhanden)."""
    if 'Rating' not in df.columns: return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Rating', palette="magma", ax=ax)

    ax.set_title("Verteilung der Sterne-Bewertungen")
    ax.set_xlabel("Sterne")
    ax.set_ylabel("Anzahl")
    return fig


def plot_aspect_heatmap(df):
    """Erstellt eine Übersicht über alle Aspekte (Positiv vs Negativ)."""
    # Wir suchen alle Spalten, die auf _pos oder _neg enden
    pos_cols = [c for c in df.columns if c.endswith('_pos')]
    neg_cols = [c for c in df.columns if c.endswith('_neg')]

    if not pos_cols: return None

    # Summen berechnen
    pos_sums = df[pos_cols].sum()
    neg_sums = df[neg_cols].sum()

    # Namen bereinigen (z.B. "Freundlichkeit_pos" -> "Freundlichkeit")
    aspect_names = [c.replace('_pos', '') for c in pos_cols]

    data = {
        'Positiv': pos_sums.values,
        'Negativ': neg_sums.values
    }
    df_aspects = pd.DataFrame(data, index=aspect_names)

    # Sortieren nach den meisten Nennungen insgesamt
    df_aspects['Total'] = df_aspects['Positiv'] + df_aspects['Negativ']
    df_aspects = df_aspects.sort_values('Total', ascending=False).drop(columns=['Total'])

    if df_aspects.empty: return None

    fig, ax = plt.subplots(figsize=(8, len(aspect_names) * 0.8 + 2))

    # Heatmap zeichnen
    sns.heatmap(df_aspects, annot=True, fmt="g", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Aspekt-Analyse: Positiv vs. Negativ Nennungen")
    return fig


def plot_bert_distribution(df):
    """Zeigt BERT-Sentiment mit Seaborn an."""
    if 'BERT_Sentiment' not in df.columns: return None

    # Wir erzwingen diese Reihenfolge, damit auch 0-Werte angezeigt werden
    categories = ["positive", "neutral", "negative"]
    counts = df['BERT_Sentiment'].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    # palette="viridis" ist der gleiche Farbstil wie bei LLaMA
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("BERT Sentiment (KI)")
    ax.set_ylabel("Anzahl Reviews")
    return fig


def plot_sentiws_distribution(df):
    """Zeigt SentiWS-Sentiment mit Seaborn an."""
    if 'Wortliste_Sentiment' not in df.columns: return None

    categories = ["positive", "neutral", "negative"]
    counts = df['Wortliste_Sentiment'].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("SentiWS Sentiment (Wortliste)")
    ax.set_ylabel("Anzahl Reviews")
    return fig