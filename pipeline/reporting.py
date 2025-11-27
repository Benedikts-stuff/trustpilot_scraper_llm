import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
sns.set_theme(style="whitegrid")


def create_wordcloud(text_series, title):
    """Erstellt eine Wordcloud aus einer Pandas-Textserie."""
    stop_words = set()
    try:
        with open("german_stopwords_full.txt", "r", encoding="utf-8") as f:
            stop_words = set(word.strip().lower() for word in f if word.strip())
    except FileNotFoundError:
        print("Warnung: Stopwords-Datei nicht gefunden")

    custom_ignore = [
        "pros", "cons", "suggestions", "verbesserungsvorschlag", "verbesserungsvorschläge",
        "gut", "schlecht", "arbeitgeber",
        "nan", "none", "null",
        "bewertung", "kommentar", "categories"
    ]

    stop_words.update(custom_ignore)

    text_data = text_series.dropna().astype(str)
    if text_data.empty: return None

    full_text = " ".join(text_data)

    filtered_words = " ".join([
        word for word in full_text.split()
        if (word.replace(":", "")).lower() not in stop_words and len(word) > 2
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
    if 'LLaMA_Kategorie' not in df.columns: return None

    categories = ["positiv", "neutral", "negativ", "Verbesserungsvorschlag"]

    s = df['LLaMA_Kategorie'].astype(str).str.lower()
    s = s.replace({"positive": "positiv", "negative": "negativ"})

    counts = s.value_counts().reindex([c.lower() for c in categories], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("LLaMA KI-Klassifizierung")
    ax.set_ylabel("Anzahl Reviews")
    ax.set_xlabel("")
    return fig


def plot_bert_distribution(df):
    if 'BERT_Sentiment' not in df.columns: return None

    categories = ["positive", "neutral", "negative"]
    counts = df['BERT_Sentiment'].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    ax.set_title("BERT Sentiment (KI)")
    ax.set_ylabel("Anzahl Reviews")
    return fig


def plot_star_distribution(df):
    if 'Rating' not in df.columns: return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Rating', palette="magma", ax=ax)

    ax.set_title("Verteilung der Sterne-Bewertungen")
    ax.set_xlabel("Sterne")
    ax.set_ylabel("Anzahl")
    return fig


def plot_aspect_heatmap(df):
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


def plot_correlation_heatmap(df):
    cols_to_corr = []
    df_calc = df.copy()

    if 'Body' in df_calc.columns:
        df_calc['Text_Length'] = df_calc['Body'].astype(str).str.len()
        cols_to_corr.append('Text_Length')

    if 'Rating' in df_calc.columns: cols_to_corr.append('Rating')
    if 'BERT_Score' in df_calc.columns: cols_to_corr.append('BERT_Score')
    if 'Wortliste_Score' in df_calc.columns: cols_to_corr.append('Wortliste_Score')

    if 'LLaMA_Kategorie' in df_calc.columns:
        llama_map = {
            "positiv": 1,
            "positive": 1,
            "neutral": 0,
            "verbesserungsvorschlag": 0,
            "negativ": -1,
            "negative": -1
        }

        df_calc['LLaMA_Score'] = df_calc['LLaMA_Kategorie'].astype(str).str.lower().map(llama_map)

        if df_calc['LLaMA_Score'].notna().any():
            cols_to_corr.append('LLaMA_Score')

    if len(cols_to_corr) < 2: return None

    # Korrelation berechnen
    corr = df_calc[cols_to_corr].corr()


    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korrelation")
    return fig


def plot_correlation_heatmap_salary(df):
    numeric_df = df.select_dtypes(include=['number'])

    numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.contains('^Unnamed')]

    if numeric_df.shape[1] < 2: return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax,
                cbar_kws={"shrink": .8})

    ax.tick_params(colors='white', which='both')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='white')

    ax.set_title("Korrelationen", color='white')
    return fig