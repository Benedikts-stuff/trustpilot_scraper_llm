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
    from nltk.corpus import stopwords
    import nltk
    try:
        stop_words = set(stopwords.words('german'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('german'))

    # Füge englische Stopwords hinzu für Trustpilot
    try:
        stop_words.update(set(stopwords.words('english')))
    except:
        pass

    # Alles zu Strings, NaNs raus
    text_data = text_series.dropna().astype(str)
    if text_data.empty: return None

    full_text = " ".join(text_data)

    filtered_words = " ".join([
        word for word in full_text.split()
        if word.lower() not in stop_words and len(word) > 2
    ])

    if not filtered_words:
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(filtered_words)

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

    # Normalisieren der Daten (Case insensitiv)
    s = df['LLaMA_Kategorie'].astype(str).str.lower()
    # Mapping auf saubere Kategorien
    s = s.replace({"positive": "positiv", "negative": "negativ"})

    counts = s.value_counts().reindex([c.lower() for c in categories], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("LLaMA KI-Klassifizierung")
    ax.set_ylabel("Anzahl Reviews")
    ax.set_xlabel("")
    return fig


def plot_bert_distribution(df):
    """Zeigt BERT-Sentiment mit Seaborn an."""
    if 'BERT_Sentiment' not in df.columns: return None

    categories = ["positive", "neutral", "negative"]
    counts = df['BERT_Sentiment'].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("BERT Sentiment (KI)")
    ax.set_ylabel("Anzahl Reviews")
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
    pos_cols = [c for c in df.columns if c.endswith('_pos')]
    neg_cols = [c for c in df.columns if c.endswith('_neg')]

    if not pos_cols: return None

    pos_sums = df[pos_cols].sum()
    neg_sums = df[neg_cols].sum()

    aspect_names = [c.replace('_pos', '') for c in pos_cols]

    data = {
        'Positiv': pos_sums.values,
        'Negativ': neg_sums.values
    }
    df_aspects = pd.DataFrame(data, index=aspect_names)
    df_aspects['Total'] = df_aspects['Positiv'] + df_aspects['Negativ']
    df_aspects = df_aspects.sort_values('Total', ascending=False).drop(columns=['Total'])

    if df_aspects.empty: return None

    fig, ax = plt.subplots(figsize=(8, len(aspect_names) * 0.8 + 2))
    sns.heatmap(df_aspects, annot=True, fmt="g", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Aspekt-Analyse: Positiv vs. Negativ Nennungen")
    return fig


# --- NEUE FUNKTIONEN FÜR ADVANCED STATS ---

def plot_correlation_heatmap(df):
    """Zeigt Korrelationen zwischen Sternen, Wortlänge und Sentiment-Scores."""
    cols_to_corr = []

    # Textlänge berechnen falls möglich
    df_calc = df.copy()
    if 'Body' in df_calc.columns:
        df_calc['Text_Length'] = df_calc['Body'].astype(str).str.len()
        cols_to_corr.append('Text_Length')

    if 'Rating' in df_calc.columns: cols_to_corr.append('Rating')
    if 'BERT_Score' in df_calc.columns: cols_to_corr.append('BERT_Score')
    if 'Wortliste_Score' in df_calc.columns: cols_to_corr.append('Wortliste_Score')

    if len(cols_to_corr) < 2: return None

    corr = df_calc[cols_to_corr].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korrelation (Zusammenhänge)")
    return fig


def plot_location_bar(df):
    """Zeigt Verteilung nach Ländern (wenn Location vorhanden)."""
    if 'Location' not in df.columns: return None

    counts = df['Location'].value_counts().head(10)  # Top 10
    if counts.empty: return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="rocket", ax=ax)
    ax.set_title("Herkunft der Bewertungen (Top 10)")
    return fig