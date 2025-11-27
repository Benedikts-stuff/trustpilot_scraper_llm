# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
import time
import json
import numpy as np

import scraper
import analysis
import reporting

st.set_page_config(
    page_title="Review Peer Analysis",
    page_icon="./growth-graph.png",
    layout="wide",
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
                .stTabs [data-baseweb="tab-list"] {
                    gap: 24px;
                }
                .stTabs [data-baseweb="tab"] {
                    height: 50px;
                    white-space: pre-wrap;
                    background-color: transparent;
                    border-radius: 4px 4px 0px 0px;
                    gap: 1px;
                    padding-top: 10px;
                    padding-bottom: 10px;
                }
        </style>
        """, unsafe_allow_html=True)

st.title("Sentiment Dashboard")
st.markdown("Zur extraktion und Analyse von Daten aus den Bewertungsportalen Kununu und Trustpilot.")

if "sources_db" not in st.session_state:
    st.session_state.sources_db = [
        {"name": "SAP Trustpilot", "url": "https://www.trustpilot.com/review/www.sap.com",
         "type": "Trustpilot (Kommentare)"},
        {"name": "SAP", "url": "https://www.kununu.com/de/sap/kommentare", "type": "Kununu (Kommentare)"},
        {"name": "Bosch", "url": "https://www.kununu.com/de/bosch-gruppe/kommentare", "type": "Kununu (Kommentare)"}
    ]

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "selected_view_sources" not in st.session_state:
    st.session_state.selected_view_sources = []


def format_category_string(cat_str: str) -> str:
    if isinstance(cat_str, dict):
        data = cat_str
    else:
        data = json.loads(cat_str)

    parts = []
    for k, v in data.items():
        score = v.get("score", "")
        text  = v.get("text", "")
        parts.append(f"{k}: {score}/5; Kommentar: {text}")
    return " \n ".join(parts)

def clean_kununu_text(df):
    if df.empty:
        return df

    parts = []

    if "Pros" in df.columns:
        parts.append("pros: " + df["Pros"].fillna(''))

    if "Cons" in df.columns:
        parts.append("cons: " + df["Cons"].fillna(''))

    if "Suggestions" in df.columns:
        parts.append("suggestions: " + df["Suggestions"].fillna(''))

    if "Categories" in df.columns:
        parts.append("categories: " + '\n' + df["Categories"].apply(format_category_string).fillna(''))

    if parts:
        df["Body"] = parts[0]
        for p in parts[1:]:
            df["Body"] += " " + p
        df["Body"] = df["Body"].str.strip()

    return df


def prepare_time_series(df, metric_type):
    """
    Wandelt den DataFrame in eine Zeitreihe mit einem einzigen numerischen Wert um.
    ROBUSTE VERSION: Behandelt Zeitzonen korrekt.
    """
    df = df.copy()

    if "Date" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce', utc=True)

    df = df.dropna(subset=["Date"])

    df["Date"] = df["Date"].dt.tz_localize(None)

    df = df.sort_values("Date")

    val_col = "Value"

    if metric_type == "BERT (Sentiment)":
        if "BERT_Sentiment" in df.columns and "BERT_Score" in df.columns:
            def get_signed_score(row):
                try:
                    label = str(row["BERT_Sentiment"]).lower()
                    score = float(row["BERT_Score"])
                    if "neg" in label: return -score
                    if "pos" in label: return score
                    return 0
                except:
                    return 0

            df[val_col] = df.apply(get_signed_score, axis=1)
        else:
            return None

    elif metric_type == "LLaMA (Klassifizierung)":
        if "LLaMA_Kategorie" in df.columns:
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



def make_distribution_chart(df, col_name, title, color_scheme="viridis"):
    if col_name not in df.columns: return None
    counts = df[col_name].value_counts().reset_index()
    counts.columns = ['Kategorie', 'Anzahl']
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Kategorie', sort='-y', title=None),
        y='Anzahl',
        color=alt.Color('Kategorie', scale=alt.Scale(scheme=color_scheme), legend=None),
        tooltip=['Kategorie', 'Anzahl']
    ).properties(title=title, height=200)
    return chart


def make_star_chart(df):
    if 'Rating' not in df.columns: return None
    counts = df['Rating'].value_counts().reset_index()
    counts.columns = ['Sterne', 'Anzahl']
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Sterne:O', sort=[1, 2, 3, 4, 5], title="Sterne"),
        y='Anzahl',
        color=alt.Color('Sterne:O', scale=alt.Scale(scheme="magma"), legend=None),
        tooltip=['Sterne', 'Anzahl']
    ).properties(title="Sterne Verteilung", height=200)
    return chart


def make_location_chart(df):
    if 'Location' not in df.columns: return None
    counts = df['Location'].value_counts().head(10).reset_index()
    counts.columns = ['Land', 'Anzahl']
    if counts.empty: return None
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Land', sort='-y', title=None),
        y='Anzahl',
        color=alt.Color('Land', scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=['Land', 'Anzahl']
    ).properties(title="Herkunft (Top 10)", height=200)
    return chart


tab_setup, tab_dashboard = st.tabs(["Setup & Daten", "Dashboard"])

# ==============================================================================
# TAB 1: SETUP
# ==============================================================================
with tab_setup:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("1. Quellen")
        with st.expander("➕ Neue Quelle", expanded=True):
            st.markdown("**URL hinzufügen**")
            new_name = st.text_input("Name", placeholder="Firma XY")
            new_type = st.selectbox("Typ", ["Trustpilot (Kommentare)", "Kununu (Kommentare)", "Kununu (Gehälter)"])
            new_url = st.text_input("URL")
            if st.button("Speichern"):
                if new_name and new_url:
                    st.session_state.sources_db.append({"name": new_name, "url": new_url, "type": new_type})
                    st.session_state.selected_view_sources.append(new_name + " " + new_type)
                    st.success("Gespeichert!")
                    st.rerun()

            st.divider()

            st.markdown("**Datei Upload**")
            uploaded_file = st.file_uploader("CSV / Excel", type=["xlsx", "csv"])
            if uploaded_file:
                f_name = f"Upload: {uploaded_file.name}"
                if not any(d['name'] == f_name for d in st.session_state.sources_db):
                    st.session_state.sources_db.append({
                        "name": f_name, "url": "UPLOADED",
                        "type": "Upload (Manuell)", "file": uploaded_file
                    })
                    st.session_state.selected_view_sources.append(f_name)
                    st.success("Datei geladen!")
                    st.rerun()

    with c2:
        st.subheader("2. Parameter & Start")

        all_sources = [(s["name"] + " " + s["type"]) for s in st.session_state.sources_db]

        default_sel = st.session_state.selected_view_sources if st.session_state.selected_view_sources else None

        sources_to_scrape = st.multiselect("Quellen auswählen", all_sources, default=default_sel)

        cp1, cp2 = st.columns(2)
        max_pages = cp1.number_input("Max. Seiten", 1, 50, 3)
        limit_analysis = cp2.number_input("Analyse Tiefe", 10, 500, 50)

        st.markdown("**Analyse-Modelle:**")
        cm1, cm2, cm3, cm4 = st.columns(4)
        use_sentiws = cm1.checkbox("Wortliste (klassisch)", True)
        use_bert = cm2.checkbox("BERT", True)
        use_spacy = cm3.checkbox("Spacy", False)
        use_llama = cm4.checkbox("LLaMA", False)

        st.divider()

        if st.button("Start: Daten extrahieren & Analysieren", type="primary", use_container_width=True):
            if not sources_to_scrape:
                st.error("Bitte wähle mindestens eine Quelle aus.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                results_buffer = st.session_state.analysis_results.copy()

                total = len(sources_to_scrape)
                for i, s_name in enumerate(sources_to_scrape):
                    status.write(f"Bearbeite: **{s_name}**...")
                    cfg = next(s for s in st.session_state.sources_db if (s["name"] + " " + s["type"])== s_name)
                    df = pd.DataFrame()

                    try:
                        # --- 1. DATEN HOLEN ---
                        if "file" in cfg:
                            f = cfg["file"]
                            if f.name.endswith('.csv'):
                                df = pd.read_csv(f)
                            else:
                                df = pd.read_excel(f)
                            df = clean_kununu_text(df)
                            col = "Body" if "Body" in df.columns else None

                        elif "Trustpilot" in cfg["type"]:
                            data = scraper.scrape_trustpilot_reviews(cfg["url"], max_pages=max_pages)
                            df = pd.DataFrame(data)
                            col = "Body"

                        elif "Kununu (Kommentare)" in cfg["type"]:
                            data = scraper.scrape_kununu_comments(cfg["url"], max_pages_to_click=max_pages)
                            df = pd.DataFrame(data)
                            df = clean_kununu_text(df)
                            col = "Body"

                        elif "Gehälter" in cfg["type"]:
                            data = scraper.scrape_kununu_salary(cfg["url"], max_pages_to_click=max_pages)
                            df = pd.DataFrame(data)
                            col = None

                    except Exception as e:
                        st.error(f"Fehler bei {s_name}: {e}")
                        continue

                    if not df.empty and col and "Salary" not in df.columns:
                        df_sub = df.head(limit_analysis).copy()

                        if use_sentiws and "Wortliste_Score" not in df_sub.columns:
                            df_sub = analysis.run_wordlist_sentiment(df_sub, col)
                        if use_bert and "BERT_Score" not in df_sub.columns:
                            df_sub = analysis.run_bert_sentiment(df_sub, col)
                        if use_llama and "LLaMA_Kategorie" not in df_sub.columns:
                            df_sub = analysis.run_llama_classification(df_sub, col)
                        if use_spacy and "lemmatized" not in df_sub.columns:
                            df_sub = analysis.run_aspect_analysis(df_sub, col)

                        results_buffer[s_name] = df_sub
                    elif not df.empty:
                        results_buffer[s_name] = df

                    progress_bar.progress((i + 1) / total)

                st.session_state.analysis_results = results_buffer
                st.session_state.selected_view_sources = sources_to_scrape
                status.success("Fertig!")
                time.sleep(1)
                st.rerun()

    st.write("---")
    st.subheader("Daten-Vorschau")
    if st.session_state.analysis_results:
        tabs = st.tabs(list(st.session_state.analysis_results.keys()))
        for t, (n, d) in zip(tabs, st.session_state.analysis_results.items()):
            with t:
                st.dataframe(d.head(100), use_container_width=True)
                st.caption(f"{len(d)} Einträge. Spalten: {list(d.columns)}")
    else:
        st.info("Keine Daten geladen.")

# ==============================================================================
# TAB 2: DASHBOARD
# ==============================================================================
with tab_dashboard:
    available = list(st.session_state.analysis_results.keys())

    cols = st.columns([1, 3])

    with cols[0].container(border=True):
        st.subheader("Einstellungen")
        valid_defs = [s for s in st.session_state.selected_view_sources if s in available]
        if not valid_defs and available: valid_defs = available[:2]

        tickers = st.multiselect("Firmen:", available, default=valid_defs)
        st.divider()

        metric_choice = st.radio("Metrik:", ["BERT (Sentiment)", "Wortliste (SentiWS)", "LLaMA (Klassifizierung)",
                                             "Sterne Bewertung", "Gehalt (Durchschnitt)"])

        st.divider()
        time_interval = st.select_slider("Zeit-Intervall (Glättung):", options=["Tag", "Woche", "Monat"], value="Woche")

    if not tickers:
        st.warning("Bitte Firmen auswählen.")
        st.stop()

    is_salary_mode = "Gehalt" in metric_choice

    combined_df = pd.DataFrame()
    salary_combined = pd.DataFrame()
    raw_data_map = {}

    resample_map = {"Tag": "D", "Woche": "W", "Monat": "M"}
    resample_code = resample_map[time_interval]

    for t in tickers:
        raw = st.session_state.analysis_results.get(t)
        if raw is not None:
            if is_salary_mode:
                if "Salary" in raw.columns and "Position" in raw.columns:
                    df_sal = raw.copy()
                    df_sal["Firma"] = t
                    if salary_combined.empty:
                        salary_combined = df_sal
                    else:
                        salary_combined = pd.concat([salary_combined, df_sal], ignore_index=True)

            else:
                ts = prepare_time_series(raw, metric_choice)
                if ts is not None:
                    ts["Source"] = t
                    if "Body" in raw.columns:
                        ts["Length"] = raw["Body"].astype(str).str.len()
                    else:
                        ts["Length"] = 0

                    if "raw" not in raw_data_map: raw_data_map["raw"] = []
                    raw_data_map["raw"].append(ts)

                    ts_agg = ts.set_index("Date")[["Value"]].resample(resample_code).mean().rename(columns={"Value": t})

                    if combined_df.empty:
                        combined_df = ts_agg
                    else:
                        combined_df = combined_df.join(ts_agg, how="outer")

    combined_smoothed = pd.DataFrame()
    if not combined_df.empty:
        combined_df = combined_df.sort_index()
        window = 7 if time_interval == "Tag" else 1
        combined_smoothed = combined_df.rolling(window=window, min_periods=1).mean()

    with cols[0].container(border=True):
        st.write("#### Trend / Übersicht")

        if is_salary_mode:
            if not salary_combined.empty:
                avg_sal = salary_combined["Salary"].mean()
                max_sal = salary_combined["Salary"].max()
                st.metric("Ø Gehalt (Alle)", f"{avg_sal:,.0f} €")
                st.metric("Max. Gehalt", f"{max_sal:,.0f} €")
            else:
                st.caption("Keine Gehaltsdaten gefunden.")
        else:
            # Bestehende Logik für Sentiment
            if not combined_smoothed.empty:
                last = combined_smoothed.last_valid_index()
                if last:
                    curr = combined_smoothed.loc[last]
                    c1, c2 = st.columns(2)
                    if not pd.isna(curr.max()):
                        c1.metric("Top", curr.idxmax(), f"{curr.max():.2f}")
                        c2.metric("Low", curr.idxmin(), f"{curr.min():.2f}", delta_color="inverse")
                else:
                    st.caption("Keine aktuellen Daten.")
            else:
                st.caption("Keine Zeitreihe.")

    with cols[1].container(border=True):

        # 1. FALL: GEHALTS-DARSTELLUNG
        if is_salary_mode:
            if not salary_combined.empty:
                st.subheader("Gehaltsvergleich nach Position")

                # 1. Berechne die tatsächliche Höhe, die das Chart braucht
                #    z.B. 40 Pixel pro Balken. Bei 50 Positionen sind das 2000 Pixel.
                num_positions = salary_combined["Position"].nunique()
                row_height = 40
                real_chart_height = max(500, num_positions * row_height + 80)

                # 2. Erstelle das "riesige" Chart
                chart = alt.Chart(salary_combined).mark_bar().encode(
                    x=alt.X("Salary:Q", title="Jahresgehalt (€)"),
                    y=alt.Y("Position:N", sort="-x", title="Position"),  # Sortiert nach Gehalt
                    color=alt.Color("Firma:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[
                        "Firma",
                        "Position",
                        alt.Tooltip("Salary", format=",.0f", title="Gehalt"),
                        alt.Tooltip("Gehaltsangaben", title="Anzahl Datensätze")
                    ]
                ).properties(
                    # WICHTIG: Hier die berechnete volle Höhe eintragen
                    height=real_chart_height,
                    title="Durchschnittsgehälter pro Position"
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Für die ausgewählten Firmen liegen keine Gehaltsdaten vor (Format: Position, Salary).")

        else:
            if not combined_smoothed.empty:
                long_df = combined_smoothed.reset_index().melt('Date', var_name='Firma', value_name='Score')
                t_title = f"Sentiment-Verlauf ({time_interval}sdurchschnitt)"
                chart = alt.Chart(long_df).mark_line(point=True).encode(
                    x=alt.X("Date:T", title="Zeit"),
                    y=alt.Y("Score:Q", title=f"Score ({metric_choice})", scale=alt.Scale(zero=False)),
                    color=alt.Color("Firma:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["Date", "Firma", alt.Tooltip("Score", format=".2f")]
                ).properties(title=t_title, height=450).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(f"Für die gewählte Metrik '{metric_choice}' sind keine Zeitdaten verfügbar.")



    if not is_salary_mode:
        st.write("---")
        st.subheader("Advanced Insights")

        if "raw" in raw_data_map:
            full_raw = pd.concat(raw_data_map["raw"])
            ac1, ac2 = st.columns(2)

            with ac1.container(border=True):
                st.markdown("**Volatilität (Streuung)**")
                base = alt.Chart(full_raw)
                box = base.mark_boxplot(extent='min-max', size=30).encode(
                    x=alt.X("Source:N", title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Value:Q", title=metric_choice),
                    color="Source:N"
                )
                points = base.mark_circle(size=15, opacity=0.3).encode(
                    x=alt.X("Source:N"),
                    y=alt.Y("Value:Q"),
                    color="Source:N",
                    xOffset="jitter:Q",
                    tooltip=["Date", "Value"]
                ).transform_calculate(jitter="random()")
                st.altair_chart(box + points, use_container_width=True)

            with ac2.container(border=True):
                st.markdown("**Länge vs. Bewertung**")
                scat = alt.Chart(full_raw).mark_circle(size=50, opacity=0.6).encode(
                    x=alt.X("Length:Q", title="Zeichenlänge"),
                    y=alt.Y("Value:Q", title=metric_choice),
                    color="Source:N",
                    tooltip=["Source", "Length", "Value"]
                ).properties(height=350).interactive()
                st.altair_chart(scat, use_container_width=True)

        st.write("---")
        st.subheader("Tiefenanalyse")

        if tickers:
            g_cols = st.columns(2)
            for i, t in enumerate(tickers):
                df = st.session_state.analysis_results.get(t)
                if df is None: continue

                with g_cols[i % 2].container(border=True):
                    st.markdown(f"### {t}")
                    t1, t2, t3, t4 = st.tabs(["Verteilung", "Wordcloud", "Korrelation", "Herkunft"])

                    with t1:
                        chart = None
                        if "LLaMA" in metric_choice and "LLaMA_Kategorie" in df.columns:
                            chart = make_distribution_chart(df, "LLaMA_Kategorie", "LLaMA Klassifizierung", "viridis")
                        elif "BERT" in metric_choice and "BERT_Sentiment" in df.columns:
                            chart = make_distribution_chart(df, "BERT_Sentiment", "BERT Sentiment", "viridis")
                        elif "Sterne" in metric_choice:
                            chart = make_star_chart(df)
                        elif "Wortliste" in metric_choice and "Wortliste_Sentiment" in df.columns:
                            chart = make_distribution_chart(df, "Wortliste_Sentiment", "SentiWS", "tealblues")

                        if chart:
                            st.altair_chart(chart, use_container_width=True)
                        elif "Salary" in df.columns:
                            st.info("Gehaltsdaten - keine Sentiment-Verteilung.")
                        else:
                            st.caption("Keine passenden Daten für die gewählte Metrik.")

                    with t2:
                        if "Body" in df.columns:
                            fig = reporting.create_wordcloud(df['Body'], f"Wordcloud: {t}")
                            if fig: st.pyplot(fig)
                        else:
                            st.info("Kein Text.")

                    with t3:
                        fig_c = reporting.plot_correlation_heatmap(df)
                        if fig_c:
                            st.pyplot(fig_c)
                        else:
                            st.info("Zu wenig Daten für Korrelation.")

                    with t4:
                        chart_loc = make_location_chart(df)
                        if chart_loc:
                            st.altair_chart(chart_loc, use_container_width=True)
                        else:
                            st.info("Keine Standortdaten verfügbar.")