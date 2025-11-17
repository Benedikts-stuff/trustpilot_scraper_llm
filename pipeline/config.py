# config.py
import os

# API-Keys
GROQ_API_KEY ='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'

# Sentiment-Wortlisten (aus word_list_sentiment.py)
SENTIWS_POS_PATH = "SentiWS_v2.0_Positive.txt"
SENTIWS_NEG_PATH = "SentiWS_v2.0_Negative.txt"
STOPWORDS_PATH = "german_stopwords_full.txt"

# Aspekt-Wörter (aus sentiment.py)
ASPECTS = {
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