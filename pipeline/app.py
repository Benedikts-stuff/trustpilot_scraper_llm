# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import time

# Importiere unsere Module
import scraper
import analysis
import reporting

# --- Seiten-Konfiguration (Wide Mode ist wichtig für diesen Look) ---
st.set_page_config(
    page_title="Review Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS HACK FÜR DEN STOCK-MARKET LOOK ---
# Das entfernt etwas Padding oben, damit es "knackiger" aussieht
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# --- INITIALISIERUNG STATE ---
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'text_column' not in st.session_state: st.session_state.text_column = "Body"


# --- HELPER: KPI BERECHNUNG ---
def calculate_kpis(df):
    kpis = {}
    # 1. Total Reviews
    kpis["count"] = len(df)

    # 2. Average Rating (Falls vorhanden)
    if "Rating" in df.columns and pd.api.types.is_numeric_dtype(df["Rating"]):
        kpis["avg_rating"] = round(df["Rating"].mean(), 2)
        kpis["delta_rating"] = None  # Hier könnte man Logik für Veränderung einbauen
    else:
        kpis["avg_rating"] = "N/A"

    # 3. Sentiment Score (Positive vs Negative Ratio)
    # Wir bevorzugen BERT, dann LLaMA, dann SentiWS
    sent_col = None
    if "BERT_Sentiment" in df.columns:
        sent_col = "BERT_Sentiment"
    elif "LLaMA_Kategorie" in df.columns:
        sent_col = "LLaMA_Kategorie"
    elif "Wortliste_Sentiment" in df.columns:
        sent_col = "Wortliste_Sentiment"

    if sent_col:
        # Wir zählen "positive" (oder "positiv")
        pos_count = df[sent_col].astype(str).str.lower().str.contains("pos").sum()
        ratio = (pos_count / len(df)) * 100
        kpis["sentiment_score"] = f"{ratio:.1f}%"
        kpis["sentiment_label"] = "Positiv-Rate"
    else:
        kpis["sentiment_score"] = "-"
        kpis["sentiment_label"] = "Sentiment"

    return kpis


# --- DISPLAY FUNKTION (IM STOCK PEERS STYLE) ---
def display_results(df, key_suffix="default"):
    # --- 1. DIE KPI REIHE (Das Herzstück des Stock Templates) ---
    kpis = calculate_kpis(df)

    # 4 Spalten für Metriken
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(label="Total Reviews", value=kpis["count"])

    with m2:
        st.metric(label="Durchschnittsbewertung", value=kpis["avg_rating"], delta=None)  # Delta könnte Trend sein

    with m3:
        st.metric(label=kpis["sentiment_label"], value=kpis["sentiment_score"])

    with m4:
        # Platzhalter für Download oder Gehalt
        if "Salary" in df.columns:
            avg_sal = f"{df['Salary'].mean():,.0f} €" if pd.api.types.is_numeric_dtype(df["Salary"]) else "-"
            st.metric(label="Ø Gehalt", value=avg_sal)
        else:
            st.metric(label="Datenquelle", value="Analysiert")

    st.markdown("---")  # Trennlinie

    # --- 2. GEHALTS-ANSICHT (Spezialfall) ---
    if "Salary" in df.columns and "Position" in df.columns:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Gehaltsverteilung (Top 15)")
            df_plot = df.sort_values("Gehaltsangaben", ascending=False).head(15)
            st.bar_chart(df_plot.set_index("Position")["Salary"])
        with c2:
            st.info("Gehaltsdaten enthalten keine Text-Reviews.")

        # Tabelle im Expander
        with st.expander("📥 Rohdaten & Tabelle anzeigen", expanded=False):
            st.dataframe(df)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Gehaelter', index=False)
            output.seek(0)
            st.download_button("Download Excel", output, "kununu_gehalt.xlsx", key=f"dl_{key_suffix}")
        return

    # --- 3. DASHBOARD VISUALISIERUNGEN ---

    # Reihe 1: Timeline (Groß, wie der Stock Chart)
    st.subheader("📈 Sentiment Trend & Verlauf")
    fig_time = reporting.plot_sentiment_timeline(df)
    if fig_time:
        st.pyplot(fig_time)
    else:
        st.info("Keine Zeitdaten für eine Trendlinie verfügbar.")

    # Reihe 2: Grid Layout (2 Spalten)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Verteilung")
        # Priorität: LLaMA > BERT > SentiWS > Sterne
        if 'LLaMA_Kategorie' in df.columns:
            fig = reporting.plot_llama_distribution(df)
            st.pyplot(fig)
        elif 'BERT_Sentiment' in df.columns:
            fig = reporting.plot_bert_distribution(df)
            st.pyplot(fig)
        elif 'Rating' in df.columns:
            fig = reporting.plot_star_distribution(df)
            st.pyplot(fig)

    with c2:
        st.subheader("Themen & Aspekte")
        fig_aspects = reporting.plot_aspect_heatmap(df)
        if fig_aspects:
            st.pyplot(fig_aspects)
        elif 'lemmatized' in df.columns:
            fig_wc = reporting.create_wordcloud(df['lemmatized'], "Top Begriffe")
            if fig_wc: st.pyplot(fig_wc)
        else:
            st.caption("Keine Aspekt-Daten verfügbar.")

    # --- 4. RAW DATA EXPANDER (Wie im Template) ---
    st.markdown("###")
    with st.expander("📥 Detaillierte Rohdaten ansehen", expanded=False):
        st.dataframe(df)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Analyse_Ergebnisse', index=False)
        output.seek(0)
        st.download_button(
            "📥 Excel Download",
            output,
            "analyse_report.xlsx",
            key=f"dl_btn_{key_suffix}"
        )


# --- SIDEBAR ---
col1, col2 = st.sidebar.columns([1, 4])
with col2:
    st.markdown("# Review Tool")

st.sidebar.header("1. Datenquelle")

source_type = st.sidebar.radio(
    "Quelle:",
    ["Trustpilot (Kommentare)", "Kununu (Kommentare)", "Kununu (Gehälter)", "Excel-Datei hochladen"],
    key="source_type"
)

url = ""
run_text_analysis_possible = True

if source_type == "Trustpilot (Kommentare)":
    url = st.sidebar.text_input("URL:", "https://www.trustpilot.com/review/...")
    st.session_state.text_column = "Body"
elif "Kununu" in source_type:
    url = st.sidebar.text_input("URL:", "https://www.kununu.com/de/...")
    if "Gehälter" in source_type:
        run_text_analysis_possible = False
    else:
        st.session_state.text_column = "Analyse_Text"
elif source_type == "Excel-Datei hochladen":
    uploaded_file = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df_upload = pd.read_excel(uploaded_file)
        st.session_state.raw_df = df_upload
        st.sidebar.success(f"Excel geladen: {len(df_upload)} Zeilen")
        cols = df_upload.columns.tolist()
        idx = cols.index("Body") if "Body" in cols else 0
        st.session_state.text_column = st.sidebar.selectbox("Text-Spalte:", cols, index=idx)

if "Excel" not in source_type:
    max_pages = st.sidebar.number_input("Max. Seiten/Klicks:", 1, 100, 5)
    if st.sidebar.button("📥 1. Daten abrufen"):
        st.session_state.results_df = None
        st.session_state.raw_df = None

        with st.spinner("Hole Daten..."):
            try:
                if "Trustpilot" in source_type:
                    data = scraper.scrape_trustpilot_reviews(url, max_pages=max_pages)
                    st.session_state.raw_df = pd.DataFrame(data)
                elif "Kommentare" in source_type:
                    data = scraper.scrape_kununu_comments(url, max_pages_to_click=max_pages)
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df['Analyse_Text'] = df['Pros'].fillna('') + ' ' + df['Cons'].fillna('') + ' ' + df[
                            'Suggestions'].fillna('')
                    st.session_state.raw_df = df
                elif "Gehälter" in source_type:
                    data = scraper.scrape_kununu_salary(url, max_pages_to_click=max_pages)
                    st.session_state.raw_df = pd.DataFrame(data)

                if st.session_state.raw_df is not None and not st.session_state.raw_df.empty:
                    st.toast(f"{len(st.session_state.raw_df)} Datensätze geladen!", icon="✅")
                else:
                    st.error("Keine Daten gefunden.")
            except Exception as e:
                st.error(f"Fehler: {e}")

st.divider()

# --- ANALYSE BUTTON ---
if run_text_analysis_possible and st.session_state.raw_df is not None:
    st.sidebar.header("2. Analyse")

    data_len = len(st.session_state.raw_df)
    default_limit = min(50, data_len)
    limit = st.sidebar.number_input("Anzahl analysieren:", 1, data_len, default_limit)

    run_ws = st.sidebar.checkbox("Wortliste (SentiWS)", True)
    run_bert = st.sidebar.checkbox("BERT (oliverguhr)", True)
    run_aspect = st.sidebar.checkbox("Aspekte (Spacy)", False)
    run_llama = st.sidebar.checkbox("LLaMA (Groq)", False)

    if st.sidebar.button("🧠 2. KI-Analyse starten"):
        df_to_analyze = st.session_state.raw_df.head(limit).copy()
        col = st.session_state.text_column

        if col not in df_to_analyze.columns:
            st.error(f"Spalte '{col}' nicht gefunden!")
        else:
            with st.spinner("KI arbeitet..."):
                try:
                    if run_ws: df_to_analyze = analysis.run_wordlist_sentiment(df_to_analyze, col)
                    if run_bert: df_to_analyze = analysis.run_bert_sentiment(df_to_analyze, col)
                    if run_aspect: df_to_analyze = analysis.run_aspect_analysis(df_to_analyze, col)
                    if run_llama: df_to_analyze = analysis.run_llama_classification(df_to_analyze, col)

                    st.session_state.results_df = df_to_analyze
                    st.toast("Analyse fertig!", icon="🎉")
                except Exception as e:
                    st.error(f"Fehler: {e}")

# --- HAUPTBEREICH (DASHBOARD) ---
# Überschrift entfernen wir hier fast, weil die Metrics oben stehen sollen
st.title(f"Analyse Report: {source_type.split('(')[0]}")

if st.session_state.results_df is not None:
    display_results(st.session_state.results_df, key_suffix="final")
elif st.session_state.raw_df is not None:
    if not run_text_analysis_possible:
        display_results(st.session_state.raw_df, key_suffix="raw")
    else:
        st.info("Rohdaten geladen. Starte links die KI-Analyse für Insights.")
        st.dataframe(st.session_state.raw_df)
else:
    st.markdown("### Willkommen!")
    st.markdown("Wähle links eine Datenquelle, um zu starten.")