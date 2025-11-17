#TODO: Fehlende kategorien ergänzen und mehr wörter finden

aspects = {
    "Freundlichkeit": {
        "pos": ["freundlich", "nett", "hilfsbereit", "zuvorkommend", "höflich", "sympathisch",
                "angenehm", "empathisch", "verständnisvoll", "menschlich", "respektvoll",
                "professionell", "herzlich", "kompetent", "aufmerksam", "ruhig", "lösungsorientiert",
                "kundenorientiert", "charmant", "locker", "entgegenkommend", "wohlwollend"
                ],
        "neg": ["unfreundlich", "patzig", "genervt", "arrogant", "herablassend", "frech",
                "kalt", "abweisend", "unhöflich", "ruppig", "schnippisch", "pampig",
                "desinteressiert", "überheblich", "respektlos", "pampig", "barsch",
                "unangemessen", "patzig", "unpersönlich", "ignorant", "kalt", "schnoddrig"
                ]
    },
    "Geschwindigkeit": {
        "pos": ["schnell", "zügig", "zeitnah", "rasch", "sofort", "prompt", "direkt", "schnellstmöglich",
                "reagiert sofort", "reaktionsschnell", "turbo", "reibungslos", "fix", "blitzschnell"
                ],
        "neg": ["langsam", "verzögert", "ewig", "wartezeit", "verzögerung", "zieht sich", "dauert lange",
            "nicht erreichbar", "reaktionszeit", "verschleppt", "nicht zeitnah", "zäh", "endlos",
            "keine rückmeldung", "warteschleife", "träge", "stillstand", "warten vergeblich"
            ]
    },
    "Kompetenz": {
        "pos": ["kompetent", "hilfreich", "sachlich", "fachkundig", "wissend", "informiert",
                "erfahren", "professionell", "lösungsorientiert", "kenntnisreich", "qualifiziert",
                "engagiert", "vertrauenswürdig", "versiert", "gewissenhaft", "routiniert"
                ],
        "neg": ["inkompetent", "unfähig", "ahnungslos", "hilflos", "überfordert", "planlos",
                "verwirrt", "unqualifiziert", "fehlerhaft", "schlecht geschult", "nicht hilfreich",
            "desinteressiert", "ratlos", "keine ahnung", "unprofessionell", "verunsichernd"
                ]
    },
}


import pandas as pd
import re
import spacy
from company_scraper_test import write_to_excel

nlp = spacy.load("de_core_news_sm")

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

df = pd.read_excel("trustpilot_reviews1.xlsx")

df['Body_clean'] = df['Body'].astype(str).str.lower()
df['lemmatized'] = df['Body_clean'].apply(lambda x: " ".join(lemmatize(x.lower())))

#df['Body_clean'] = df['Body'].str.lower().str.replace(r'[^a-zäöüß ]', '', regex=True)

def count_aspect_sentiment(text, aspect_words):
    score = {"pos": 0, "neg": 0}
    for label in ["pos", "neg"]:
        for word in aspect_words[label]:
            if word in text:
                score[label] += 1
    return score

for aspect in aspects.keys():
    df[f"{aspect}_pos"] = 0
    df[f"{aspect}_neg"] = 0

for idx, row in df.iterrows():
    text = row['lemmatized']
    for aspect, words in aspects.items():
        scores = count_aspect_sentiment(text, words)
        df.at[idx, f"{aspect}_pos"] = scores["pos"]
        df.at[idx, f"{aspect}_neg"] = scores["neg"]


# Basisfelder, die du behalten willst
base_cols = ['Date', 'Body', 'lemmatized']

# Dynamisch alle Aspekt-Score-Spalten finden
aspect_cols = [col for col in df.columns if any(key in col for key in aspects.keys())]
totals = df[aspect_cols].sum().astype(int)
summary = totals.rename_axis("Aspect_Sentiment").reset_index(name="Anzahl")
summary[["Aspect", "Sentiment"]] = summary["Aspect_Sentiment"].str.rsplit("_", n=1, expand=True)
pivot = summary.pivot(index="Aspect", columns="Sentiment", values="Anzahl").reset_index()
pivot.insert(1, '', '')  # Leerspalte
pivot = pivot[["Aspect", "", "pos", "neg"]]
print(pivot)
write_to_excel("aspekt_sentiment_summary_formatiert.xlsx","Sheet1", pivot)
# Zusammenführen
export_cols = base_cols + aspect_cols

# Neuer DataFrame
df_aspects = df[export_cols]

df_grouped = pd.DataFrame
numeric_cols = df_aspects.select_dtypes(include="number")
summary = numeric_cols.sum().astype(int)
df_aspects.to_excel("aspekt_sentiment.xlsx", index=False)
