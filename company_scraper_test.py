from scraper import scrape_trustpilot_reviews, get_overall_rating
import pandas as pd
import os
from openpyxl import load_workbook
from openpyxl.styles import numbers
from datetime import datetime
from pandas.tseries.offsets import MonthEnd


def trustpilot_trend(url='https://de.trustpilot.com/review/ergo-reiseversicherung.de?date=last6months', output_file="trustpilot_summary_transposed.xlsx" ):
    # 1. Trustpilot Reviews scrapen
    base_url = url
    reviews = scrape_trustpilot_reviews(base_url)

    # 2. DataFrame aus Reviews
    df = pd.DataFrame(reviews)
    print(df.columns)

    # 3. Datum konvertieren und Monats-Spalte extrahieren
    df["Date"] = pd.to_datetime(df["Date"])  # Wandelt ISO-String zu echtem Datum
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    # Heute
    today = pd.Timestamp.today()

    # Letzter Tag des Vormonats
    cutoff = today.replace(day=1) - pd.Timedelta(days=1)

    # Nur Reviews bis einschließlich letzten vollständigen Monat behalten
    df = df[df["Date"] <= cutoff]

    # Monats-Spalte wie gehabt
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    # 4. Hauptdaten-Excel speichern (Datum wird korrekt geschrieben, aber Format muss gesetzt werden)
    main_file = "trustpilot_reviews1.xlsx"
    df.to_excel(main_file, index=False)

    # 6. Gruppieren nach Monat
    grouped = df.groupby('Month')
    summary = pd.DataFrame()
    summary['1_2_Stars'] = grouped['Rating'].apply(lambda x: ((x <= 2).sum()))
    summary['3_Stars'] = grouped['Rating'].apply(lambda x: ((x == 3).sum()))
    summary['4_5_Stars'] = grouped['Rating'].apply(lambda x: ((x >= 4).sum()))
    summary['Total'] = summary.sum(axis=1)
    summary['PosRate'] = round((summary['4_5_Stars'] / summary['Total']) * 100, 2)

    # 7. Transponieren für Übersicht
    summary_transposed = summary[['1_2_Stars', '3_Stars', '4_5_Stars', 'PosRate']].T
    summary_transposed.columns.name = None
    summary_transposed = summary_transposed.sort_index(axis=1)

    # 8. Zusammenfassungsdatei schreiben oder ersetzen
    summary_file = output_file
    if os.path.exists(summary_file):
        writer = pd.ExcelWriter(summary_file, engine='openpyxl', mode='a', if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(summary_file, engine='openpyxl', mode='w')

    with writer:
        summary_transposed.to_excel(writer, sheet_name='Sheet1')

def trustpilot_score(url='https://de.trustpilot.com/review/c24.de', number_reviews_summary='trustpilot_general_score.xlsx'):
    trust_score, rating_title, total_ratings, stars, stars_absolute = get_overall_rating(url)

    star_order = {"five": 5, "four": 4, "three": 3, "two": 2, "one": 1}
    data = []
    for key in sorted(stars_absolute.keys(), key=lambda k: -star_order[k]):
        data.append([star_order[key], "", stars_absolute[key]])

    stars_absolute_df = pd.DataFrame(data, columns=["Sterne", " ", "Anzahl"])

    if os.path.exists(number_reviews_summary):
        writer = pd.ExcelWriter(number_reviews_summary, engine='openpyxl', mode='a', if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(number_reviews_summary, engine='openpyxl', mode='w')

    with writer:
        stars_absolute_df.to_excel(writer, sheet_name=number_reviews_summary, index=False)

trustpilot_score()