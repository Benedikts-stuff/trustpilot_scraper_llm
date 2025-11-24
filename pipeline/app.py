# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
import time
import numpy as np
from datetime import datetime

# Importiere deine lokalen Module
import scraper
import analysis
import reporting

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Review Peer Analysis",
    page_icon="📈",
    layout="wide",
)

# --- CSS VOM STOCK PEER TEMPLATE ---
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

st.title("📊 Review & Sentiment Peer Analysis")
st.markdown("Vergleiche Arbeitgeber-Bewertungen und Stimmungen.")

# --- STATE MANAGEMENT ---
if "sources_db" not in st.session_state:
    st.session_state.sources_db = [
        {"name": "Beispiel Trustpilot", "url": "https://www.trustpilot.com/review/www.sap.com",
         "type": "Trustpilot (Kommentare)"},
        {"name": "Beispiel Kununu", "url": "https://www.kununu.com/de/telekom", "type": "Kununu (Kommentare)"}
    ]

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "selected_view_sources" not in st.session_state:
    st.session_state.selected_view_sources = []


# --- HELPER: SCORE BERECHNUNG ---
def prepare_time_series(df, metric_type):
    """
    Wandelt den DataFrame in eine Zeitreihe mit einem einzigen numerischen Wert um,
    basierend auf der gewählten Metrik.
    """
    df = df.copy()

    # 1. Datum parsen
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # 2. Metrik berechnen
    val_col = "Value"

    if metric_type == "BERT (Sentiment)":
        if "BERT_Sentiment" in df.columns and "BERT_Score" in df.columns:
            # Wir machen aus Label + Score einen Wert zwischen -1 und 1
            # Logik: Score * (1 wenn positiv, -1 wenn negativ)
            def get_signed_score(row):
                label = str(row["BERT_Sentiment"]).lower()
                score = float(row["BERT_Score"])
                if "neg" in label: return -score
                if "pos" in label: return score
                return 0  # neutral

            df[val_col] = df.apply(get_signed_score, axis=1)
        else:
            return None

    elif metric_type == "LLaMA (Klassifizierung)":
        if "LLaMA_Kategorie" in df.columns:
            # Mapping: positiv=1, negativ=-1, neutral=0, verbesserung=0.5 (optional)
            mapping = {"positiv": 1.0, "positive": 1.0, "negativ": -1.0, "negative": -1.0, "neutral": 0.0}
            df[val_col] = df["LLaMA_Kategorie"].astype(str).str.lower().map(mapping).fillna(0.0)
        else:
            return None

    elif metric_type == "Wortliste (SentiWS)":
        if "Wortliste_Score" in df.columns:
            df[val_col] = pd.to_numeric(df["Wortliste_Score"], errors='coerce').fillna(0)
        else:
            return None

    elif metric_type == "Sterne Bewertung":
        if "Rating" in df.columns:
            df[val_col] = pd.to_numeric(df["Rating"], errors='coerce')
        else:
            return None

    else:
        return None

    return df[["Date", val_col]]


# --- TABS STRUKTUR ---
tab_setup, tab_dashboard = st.tabs(["🛠️ Setup & Daten", "📈 Analyse-Dashboard"])

# ==============================================================================
# TAB 1: DATENERFASSUNG & EINSTELLUNGEN
# ==============================================================================
with tab_setup:
    c_conf1, c_conf2 = st.columns([1, 2])

    with c_conf1:
        st.subheader("1. Quellen verwalten")

        # Bestehende Quellen anzeigen
        st.write("**Gespeicherte Quellen:**")
        for s in st.session_state.sources_db:
            st.text(f"• {s['name']} ({s['type']})")

        with st.expander("➕ Neue Quelle hinzufügen", expanded=True):
            new_name = st.text_input("Name (Firma)", placeholder="z.B. SAP")
            new_type = st.selectbox("Typ", ["Trustpilot (Kommentare)", "Kununu (Kommentare)", "Kununu (Gehälter)"])
            new_url = st.text_input("URL")
            if st.button("Speichern"):
                if new_name and new_url:
                    st.session_state.sources_db.append({"name": new_name, "url": new_url, "type": new_type})
                    st.success(f"{new_name} hinzugefügt!")
                    st.rerun()

    with c_conf2:
        st.subheader("2. Scraping & Analyse Parameter")

        # Auswahl was gescrapt werden soll (Multiselect aus DB)
        source_options = [s["name"] for s in st.session_state.sources_db]
        sources_to_scrape = st.multiselect("Welche Quellen sollen aktualisiert/analysiert werden?",
                                           source_options, default=source_options[:1] if source_options else None)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Scraping Limits**")
            max_pages = st.number_input("Max. Seiten scrapen", 1, 50, 3)

        with col_p2:
            st.markdown("**Analyse Tiefe**")
            limit_analysis = st.number_input("Anzahl Reviews analysieren", 10, 500, 50)

        st.markdown("**KI-Modelle auswählen**")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        use_sentiws = c_m1.checkbox("Wortliste (SentiWS)", True)
        use_bert = c_m2.checkbox("BERT (Sentiment)", True)
        use_spacy = c_m3.checkbox("Spacy (Aspekte)", False)
        use_llama = c_m4.checkbox("LLaMA (Groq)", False)

        st.markdown("---")

        if st.button("🚀 Daten abrufen & Analyse starten", type="primary"):
            if not sources_to_scrape:
                st.error("Bitte wähle mindestens eine Quelle aus.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                results_buffer = st.session_state.analysis_results.copy()  # Behalte alte Ergebnisse

                total_steps = len(sources_to_scrape)
                for i, s_name in enumerate(sources_to_scrape):
                    status.write(f"Bearbeite: **{s_name}**...")

                    # Config finden
                    cfg = next(s for s in st.session_state.sources_db if s["name"] == s_name)

                    # 1. Scraping
                    df = pd.DataFrame()
                    try:
                        if "Trustpilot" in cfg["type"]:
                            data = scraper.scrape_trustpilot_reviews(cfg["url"], max_pages=max_pages)
                            df = pd.DataFrame(data)
                            col = "Body"
                        elif "Kununu (Kommentare)" in cfg["type"]:
                            data = scraper.scrape_kununu_comments(cfg["url"], max_pages_to_click=max_pages)
                            df = pd.DataFrame(data)
                            if not df.empty:
                                df['Body'] = df['Pros'].fillna('') + ' ' + df['Cons'].fillna('')
                            col = "Body"
                        elif "Gehälter" in cfg["type"]:
                            data = scraper.scrape_kununu_salary(cfg["url"], max_pages_to_click=max_pages)
                            df = pd.DataFrame(data)
                            col = None
                    except Exception as e:
                        st.error(f"Fehler bei {s_name}: {e}")
                        continue

                    # 2. Analyse
                    if not df.empty and col and "Salary" not in df.columns:
                        df_sub = df.head(limit_analysis).copy()
                        if use_sentiws: df_sub = analysis.run_wordlist_sentiment(df_sub, col)
                        if use_bert: df_sub = analysis.run_bert_sentiment(df_sub, col)
                        if use_llama: df_sub = analysis.run_llama_classification(df_sub, col)
                        if use_spacy: df_sub = analysis.run_aspect_analysis(df_sub, col)
                        results_buffer[s_name] = df_sub
                    elif not df.empty:
                        results_buffer[s_name] = df

                    progress_bar.progress((i + 1) / total_steps)

                st.session_state.analysis_results = results_buffer
                st.session_state.selected_view_sources = sources_to_scrape  # Setze Default Auswahl im Dashboard
                status.success("Fertig! Wechsel zum Tab 'Analyse-Dashboard' um die Ergebnisse zu sehen.")
                time.sleep(1)
                st.rerun()

# ==============================================================================
# TAB 2: STOCK PEER DASHBOARD
# ==============================================================================
with tab_dashboard:
    available_sources = list(st.session_state.analysis_results.keys())

    # 1. Spalten-Layout (1:3)
    cols = st.columns([1, 3])

    # --- LINKES PANEL (CONTROLS) ---
    top_left_cell = cols[0].container(border=True)

    with top_left_cell:
        st.subheader("Einstellungen")

        # A) Firmen Auswahl
        tickers = st.multiselect(
            "1. Firmen vergleichen:",
            options=available_sources,
            default=st.session_state.selected_view_sources if st.session_state.selected_view_sources else available_sources[
                                                                                                          :2],
            placeholder="Wähle Firmen..."
        )

        st.write("")

        # B) Metrik Auswahl (NEU: Basierend auf welchem Output?)
        metric_choice = st.radio(
            "2. Daten-Basis für Vergleich:",
            ["BERT (Sentiment)", "Wortliste (SentiWS)", "LLaMA (Klassifizierung)", "Sterne Bewertung"],
            index=0
        )

        st.caption(
            "Hinweis: 'BERT' und 'Wortliste' zeigen Werte von ca. -1 (negativ) bis +1 (positiv). 'Sterne' zeigt 1-5.")

    if not tickers:
        st.warning("Bitte wähle mindestens eine Firma aus.")
        st.stop()

    # --- DATEN VORBEREITUNG (COMBINED DF) ---
    # Wir erstellen ein DataFrame, das für ALLE gewählten Firmen die Zeitreihe enthält.
    # Wichtig: Wir resamplen auf Tage, um Vergleichbarkeit zu gewährleisten.

    combined_df = pd.DataFrame()

    for t in tickers:
        raw_df = st.session_state.analysis_results.get(t)
        if raw_df is not None:
            ts_df = prepare_time_series(raw_df, metric_choice)

            if ts_df is not None and not ts_df.empty:
                # Umbenennen für Join
                ts_df = ts_df.rename(columns={"Value": t})
                ts_df = ts_df.set_index("Date")

                # Resampling auf Tagesbasis (Mittelwert pro Tag), damit Index eindeutig ist
                ts_df = ts_df.resample('D').mean()

                if combined_df.empty:
                    combined_df = ts_df
                else:
                    combined_df = combined_df.join(ts_df, how="outer")

    # Sortieren und Interpolieren (optional, hier lassen wir Lücken ggf. sichtbar oder füllen sie für den Plot)
    if not combined_df.empty:
        combined_df = combined_df.sort_index()
        # Gleitender Durchschnitt für Glättung (macht Charts lesbarer)
        combined_df_smoothed = combined_df.rolling(window=7, min_periods=1).mean()
    else:
        combined_df_smoothed = pd.DataFrame()

    # --- METRICS (LINKS UNTEN) ---
    bottom_left_cell = cols[0].container(border=True)
    with bottom_left_cell:
        if not combined_df_smoothed.empty:
            # Letzter verfügbarer Wert (Stand heute/letztes Datum)
            last_valid_idx = combined_df_smoothed.last_valid_index()
            if last_valid_idx:
                current_vals = combined_df_smoothed.loc[last_valid_idx]

                best_ticker = current_vals.idxmax()
                worst_ticker = current_vals.idxmin()

                c_met1, c_met2 = st.columns(2)

                val_fmt = "{:.2f}"
                c_met1.metric("Top Trend", best_ticker, val_fmt.format(current_vals[best_ticker]))
                c_met2.metric("Low Trend", worst_ticker, val_fmt.format(current_vals[worst_ticker]),
                              delta_color="inverse")
            else:
                st.info("Daten vorhanden, aber keine aktuellen Werte.")
        else:
            st.info("Keine Daten für die gewählte Metrik.")

    # --- MAIN CHART (RECHTS) - ZEITLICHER VERLAUF ---
    right_cell = cols[1].container(border=True)
    with right_cell:
        if not combined_df_smoothed.empty:
            # Für Altair in Long-Format bringen
            long_df = combined_df_smoothed.reset_index().melt('Date', var_name='Firma', value_name='Score')

            # Titel basierend auf Metrik
            chart_title = f"Zeitlicher Verlauf: {metric_choice}"

            chart = alt.Chart(long_df).mark_line(point=True).encode(
                x=alt.X("Date:T", title="Datum"),
                y=alt.Y("Score:Q", title="Score (geglättet)", scale=alt.Scale(zero=False)),
                color=alt.Color("Firma:N", legend=alt.Legend(orient="bottom")),
                tooltip=["Date", "Firma", alt.Tooltip("Score", format=".2f")]
            ).properties(
                title=chart_title,
                height=450
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning(f"Für die ausgewählten Firmen gibt es keine Daten im Bereich '{metric_choice}'.")

    st.write("")

    # --- INDIVIDUAL VS PEER AVERAGE (GRID) ---
    if not combined_df_smoothed.empty and len(tickers) >= 2:
        st.markdown("### 🆚 Vergleich: Firma vs. Durchschnitt (Peers)")
        st.markdown("Wie schneidet die Firma im Vergleich zum Durchschnitt der anderen ab?")

        NUM_COLS = 4
        grid_cols = st.columns(NUM_COLS)

        # Nutze geglättete Daten für den Vergleich
        df_comp = combined_df_smoothed.dropna(how='all')

        for i, ticker in enumerate(tickers):
            if ticker not in df_comp.columns: continue

            # Peer Average berechnen (Durchschnitt aller ANDEREN Spalten)
            other_cols = [c for c in df_comp.columns if c != ticker]
            if not other_cols: continue

            peer_avg = df_comp[other_cols].mean(axis=1)

            # Plot Data Vorbereitung
            plot_data = pd.DataFrame({
                "Date": df_comp.index,
                ticker: df_comp[ticker],
                "Peer Average": peer_avg
            }).melt(id_vars=["Date"], var_name="Type", value_name="Value")

            # 1. Line Chart
            line_chart = alt.Chart(plot_data).mark_line().encode(
                x=alt.X("Date:T", axis=alt.Axis(labels=False, title=None)),
                y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=None),
                color=alt.Color("Type:N",
                                scale=alt.Scale(domain=[ticker, "Peer Average"], range=["#1f77b4", "#d62728"]),
                                legend=None),
                tooltip=["Date", "Type", alt.Tooltip("Value", format=".2f")]
            ).properties(title=f"{ticker} vs. Ø", height=180)

            # 2. Delta Area Chart
            delta_data = pd.DataFrame({
                "Date": df_comp.index,
                "Delta": df_comp[ticker] - peer_avg
            })

            area_chart = alt.Chart(delta_data).mark_area(opacity=0.6).encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Delta:Q", title="Abweichung"),
                color=alt.condition(
                    alt.datum.Delta > 0,
                    alt.value("green"),
                    alt.value("red")
                ),
                tooltip=[alt.Tooltip("Date", format="%Y-%m-%d"), alt.Tooltip("Delta", format=".2f")]
            ).properties(height=180)

            # Anzeigen im Grid
            # Spalte 1: Line Chart
            c_idx = (i * 2) % NUM_COLS
            with grid_cols[c_idx].container(border=True):
                st.altair_chart(line_chart, use_container_width=True)

            # Spalte 2: Delta Chart
            with grid_cols[c_idx + 1].container(border=True):
                st.altair_chart(area_chart, use_container_width=True)

    elif len(tickers) < 2 and not combined_df_smoothed.empty:
        st.info("Wähle mindestens 2 Firmen für den Peer-Vergleich.")

    # --- RAW DATA ---
    st.write("---")
    with st.expander("📥 Detaillierte Daten ansehen"):
        st.dataframe(combined_df, use_container_width=True)