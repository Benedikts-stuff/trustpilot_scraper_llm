# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import time

# Importiere unsere eigenen Module
import scraper
import analysis
import reporting

# import config (config.py wird nicht mehr gebraucht, wenn Keys in analysis.py sind)

# --- Seiten-Konfiguration ---
st.set_page_config(
    page_title="Review Analyse Tool",
    page_icon="📊",
    layout="wide"
)


# --- HILFSFUNKTION FÜR PUNKT 3 (STATE) ---
def display_results(df):
    """Zeigt die Ergebnisse an, die im Session State gespeichert sind."""

    # Prüfen, ob es ein Analyse- oder Gehalts-DF ist
    is_salary_df = "Salary" in df.columns and "Position" in df.columns

    if is_salary_df:
        st.header("Ergebnis-Datenbank (Gehälter)")
        st.dataframe(df)

        st.subheader("Gehaltsverteilung (Top 15 Positionen nach Anzahl)")
        df_plot = df.sort_values("Gehaltsangaben", ascending=False).head(15)
        st.bar_chart(df_plot.set_index("Position")["Salary"],
                     x_label="Position", y_label="Durchschnittsgehalt (€)")

        # Download-Button (Gehälter)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Gehaelter', index=False)
        output.seek(0)

        st.download_button(
            label="📥 Gehalts-Daten als Excel herunterladen",
            data=output,
            file_name=f"kununu_gehaelter_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        # Standard-Textanalyse-Ansicht
        st.header("Ergebnis-Datenbank (Text-Analyse)")
        st.dataframe(df)

        st.header("Visualisierungen")
        col1, col2 = st.columns(2)
        if 'BERT_Sentiment' in df.columns:
            with col1:
                st.subheader("BERT Sentiment Verteilung")
                st.bar_chart(df['BERT_Sentiment'].value_counts())
        if 'LLaMA_Kategorie' in df.columns:
            with col2:
                st.subheader("LLaMA Kategorie Verteilung")
                st.bar_chart(df['LLaMA_Kategorie'].value_counts())
        if 'lemmatized' in df.columns:
            st.subheader("Wordcloud (aus Aspekt-Analyse)")
            fig_wc = reporting.create_wordcloud(df['lemmatized'], "Häufigste Begriffe")
            if fig_wc: st.pyplot(fig_wc)

        # Download-Button (Text)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Analyse_Ergebnisse', index=False)
        output.seek(0)

        st.download_button(
            label="📥 Analyse-Ergebnisse als Excel herunterladen",
            data=output,
            file_name=f"review_analyse_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# --- INITIALISIERUNG VON PUNKT 3 (STATE) ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- 1. Sidebar (Eingaben & Steuerung) ---
st.sidebar.title("📊 Review Analyse Tool")
st.sidebar.header("1. Datenquelle auswählen")

source_type = st.sidebar.radio(
    "Woher kommen die Daten?",
    [
        "Trustpilot (Kommentare)",
        "Kununu (Kommentare)",
        "Kununu (Gehälter)",
        "Excel-Datei hochladen"
    ],
    key="source_type",
    # Wenn eine neue Quelle gewählt wird, setze den alten DF zurück
    on_change=lambda: st.session_state.update(results_df=None)
)

# Platzhalter
df = None
url = ""
text_column = "Body"
run_text_analysis = True

if source_type == "Trustpilot (Kommentare)":
    url = st.sidebar.text_input("Trustpilot URL:", "https://www.trustpilot.com/review/...")
    text_column = "Body"
elif source_type == "Kununu (Kommentare)":
    url = st.sidebar.text_input("Kununu URL:", "https://www.kununu.com/de/firmenname/kommentare")
    text_column = "Analyse_Text"  # Diese Spalte erstellen wir künstlich
elif source_type == "Kununu (Gehälter)":
    url = st.sidebar.text_input("Kununu URL:", "https://www.kununu.com/de/firmenname/gehalt")
    run_text_analysis = False  # Keine Textanalyse für Gehaltsdaten
elif source_type == "Excel-Datei hochladen":
    uploaded_file = st.sidebar.file_uploader("Excel-Datei auswählen (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.sidebar.info(f"Datei geladen: {len(df)} Zeilen.")
        available_columns = df.columns.tolist()
        text_column = st.sidebar.selectbox(
            "Welche Spalte enthält den Review-Text?",
            available_columns,
            index=available_columns.index("Body") if "Body" in available_columns else 0
        )

st.sidebar.header("2. Analyse-Optionen")

# NEU: PUNKT 1 (Scrape-Limit)
if "Excel" not in source_type:
    max_pages_scrape = st.sidebar.number_input(
        "Max. Seiten/Klicks scrapen:",
        min_value=1, value=5, step=1,
        help="Limitiert, wie viele 'Mehr anzeigen'-Seiten (Trustpilot) oder Klicks (Kununu) geladen werden."
    )
else:
    max_pages_scrape = 0  # Nicht relevant für Excel-Upload

# Nur Textanalyse-Optionen anzeigen, wenn relevant
if run_text_analysis:
    limit_reviews_analyze = st.sidebar.number_input(
        "Max. Reviews analysieren:",
        min_value=1, value=50, step=10,
        help="Limitiert die Analyse auf die ersten X Reviews *nach* dem Scrapen."
    )
    st.sidebar.subheader("Analysen auswählen:")
    run_wordlist = st.sidebar.checkbox("Wortlisten-Sentiment (SentiWS)", value=True)
    run_bert = st.sidebar.checkbox("BERT-Sentiment (oliverguhr)", value=True)
    run_aspect = st.sidebar.checkbox("Aspekt-Analyse (Spacy)", value=False)
    run_llama = st.sidebar.checkbox("LLaMA-Klassifizierung (Groq)", value=False,
                                    help="SEHR LANGSAM! (ca. 3-5 Sek. pro Review)")
else:
    st.sidebar.info("Für Gehaltsdaten ist keine Textanalyse verfügbar.")
    limit_reviews_analyze = 0
    run_wordlist = run_bert = run_aspect = run_llama = False

st.sidebar.header("3. Analyse starten")
start_button = st.sidebar.button("🚀 Analyse jetzt starten")

# --- 2. Hauptbereich (Logik & Ergebnisse) ---
st.title("Analyse-Ergebnisse")

if start_button:
    # Wenn der Start-Button gedrückt wird, ALLES zurücksetzen
    st.session_state.results_df = None

    # --- Schritt A: Daten laden ---
    if source_type == "Trustpilot (Kommentare)":
        with st.spinner(f"Scrape {max_pages_scrape} Trustpilot-Seiten..."):
            reviews_list = scraper.scrape_trustpilot_reviews(url, max_pages=max_pages_scrape)
            if not reviews_list: st.error("Keine Reviews gefunden."); st.stop()
            df = pd.DataFrame(reviews_list)

    elif source_type == "Kununu (Kommentare)":
        with st.spinner(f"Scrape Kununu Reviews (max. {max_pages_scrape} Klicks)... (Selenium startet)"):
            reviews_list = scraper.scrape_kununu_comments(url, max_pages_to_click=max_pages_scrape)
            if not reviews_list: st.error("Keine Reviews gefunden."); st.stop()
            df = pd.DataFrame(reviews_list)
            # WICHTIG: Text für Analyse zusammenführen
            df['Analyse_Text'] = df['Pros'].fillna('') + ' ' + df['Cons'].fillna('') + ' ' + df['Suggestions'].fillna(
                '')
            text_column = "Analyse_Text"

    elif source_type == "Kununu (Gehälter)":
        with st.spinner(f"Scrape Kununu Gehälter (max. {max_pages_scrape} Klicks)... (Selenium startet)"):
            salary_list = scraper.scrape_kununu_salary(url, max_pages_to_click=max_pages_scrape)
            if not salary_list: st.error("Keine Gehälter gefunden."); st.stop()
            df = pd.DataFrame(salary_list)

    # (Daten aus Excel-Upload sind bereits in 'df' geladen)

    # --- Schritt B: Prüfen, ob Daten vorhanden sind ---
    if df is None or df.empty:
        st.error("Bitte zuerst eine Datenquelle (URL oder Datei) angeben.")
        st.stop()

    # --- Schritt C: Text-Analyse-Pipeline ---
    if run_text_analysis:
        if text_column not in df.columns:
            st.error(f"Text-Spalte '{text_column}' nicht gefunden.");
            st.stop()

        df_original_len = len(df)
        df_analysis = df.head(limit_reviews_analyze).copy()

        st.info(f"Analyse gestartet für die **ersten {len(df_analysis)}** von {df_original_len} Reviews.")

        try:
            if run_wordlist:
                with st.spinner("Läuft: Wortlisten-Analyse (SentiWS)..."):
                    df_analysis = analysis.run_wordlist_sentiment(df_analysis, text_column)
            if run_bert:
                with st.spinner("Läuft: BERT-Sentiment-Analyse..."):
                    df_analysis = analysis.run_bert_sentiment(df_analysis, text_column)
            if run_aspect:
                with st.spinner("Läuft: Aspekt-Analyse (Spacy)..."):
                    df_analysis = analysis.run_aspect_analysis(df_analysis, text_column)
            if run_llama:
                st.warning("LLaMA-Analyse gestartet. Dies kann SEHR lange dauern...")
                with st.spinner("Läuft: LLaMA-Klassifizierung (SEHR LANGSAM)..."):
                    df_analysis = analysis.run_llama_classification(df_analysis, text_column)

            st.success("Text-Analyse abgeschlossen.")
            # HIER IST PUNKT 3: Ergebnisse im State speichern
            st.session_state.results_df = df_analysis.copy()

        except Exception as e:
            st.error(f"Ein Fehler bei der Text-Analyse ist aufgetreten: {e}")
            st.exception(e)

    else:
        # HIER IST PUNKT 3: Ergebnisse im State speichern (für Gehälter)
        st.session_state.results_df = df.copy()

# --- SCHRITT 3: ERGEBNISSE IMMER ANZEIGEN (PUNKT 3) ---
# Dieser Block wird *immer* ausgeführt, auch nach einem Download-Klick.
# Wenn Daten im State sind, werden sie angezeigt.
if st.session_state.results_df is not None:
    display_results(st.session_state.results_df)