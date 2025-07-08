from scraper import scrape_trustpilot_reviews, get_overall_rating
import pandas as pd
import os
import time
from openpyxl import load_workbook
from openpyxl.styles import numbers
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
import win32com.client as win32

def write_to_excel(filepath, sheet_name, df):
    from openpyxl import load_workbook

    if os.path.exists(filepath):
        # Excel-Datei existiert -> anhängen
        book = load_workbook(filepath)

        if sheet_name in book.sheetnames:
            sheet = book[sheet_name]
            sheet.delete_rows(1, sheet.max_row)
        else:
            pass

        # Writer mit bestehendem Workbook verbinden
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            writer._book = book  # Undokumentiert, aber funktional
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    else:
        # Datei existiert noch nicht → neu anlegen
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def trustpilot_trend(url='https://de.trustpilot.com/review/ergo-reiseversicherung.de?date=last6months', output_file="trustpilot_summary.xlsx" ):
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
    summary_transposed = summary[['PosRate','1_2_Stars', '4_5_Stars', '3_Stars']].T
    summary_transposed.columns.name = None
    summary_transposed = summary_transposed.sort_index(axis=1)

    # Leere Zeile mit leerem Index
    empty_row = pd.DataFrame([[""] * summary_transposed.shape[1]],
                             columns=summary_transposed.columns,
                             index=[""])  # leerer Index

    # Leere Zeile einfügen, z.B. an 2. Stelle
    top = summary_transposed.iloc[:0]
    bottom = summary_transposed.iloc[0:]
    summary_with_empty = pd.concat([top, empty_row, bottom])
    summary_transposed = summary_with_empty
    # 8. Zusammenfassungsdatei schreiben oder ersetzen
    summary_file = output_file
    write_to_excel(summary_file, summary_file, summary_transposed)

    trigger_thinkcell_update('C:\\Users\\beng\\PycharmProjects\\trustpilot_scraper_llm\\trustpilot_summary.xlsx')

def trigger_thinkcell_update(excel_path):
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    excel.Visible = True  # Nicht anzeigen

    workbook = excel.Workbooks.Open(excel_path)
    #time.sleep(30)
    #workbook.Save()        # Speichern (triggert Update)
    #workbook.Close(False)  # Schließen ohne weitere Änderungen

    #excel.Quit()

def trustpilot_score(url='https://de.trustpilot.com/review/finn.com', number_reviews_summary='trustpilot_general_score.xlsx'):
    trust_score, rating_title, total_ratings, stars, stars_absolute = get_overall_rating(url)

    star_order = {"five": 5, "four": 4, "three": 3, "two": 2, "one": 1}
    data = []
    for key in sorted(stars_absolute.keys(), key=lambda k: -star_order[k]):
        data.append([star_order[key], "", stars_absolute[key]])

    stars_absolute_df = pd.DataFrame(data, columns=["Sterne", " ", "Anzahl"])

    write_to_excel(number_reviews_summary, number_reviews_summary, stars_absolute_df)
    trigger_thinkcell_update('C:\\Users\\beng\\PycharmProjects\\trustpilot_scraper_llm\\trustpilot_general_score.xlsx')


time_intervall = "?date=last6months"
url = "https://de.trustpilot.com/review/www.amazon.de"
#trustpilot_score(url=url)
trustpilot_trend(url=url+time_intervall)