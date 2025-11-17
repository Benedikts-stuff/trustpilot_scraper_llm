import re

def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f if word.strip())

stop_words = load_stopwords("german_stopwords_full.txt")

def load_sentiws(filepath):
    word_scores = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lemma_pos = parts[0]  # z.B. Abbau|NN
                score = float(parts[1])

                # Lemma selbst ohne POS speichern
                lemma = lemma_pos.split("|")[0]
                word_scores[lemma.lower()] = score

                # Falls Flexionsformen existieren
                if len(parts) == 3:
                    forms = parts[2].split(",")
                    for form in forms:
                        word_scores[form.lower()] = score
    return word_scores

pos_words = load_sentiws("SentiWS_v2.0_Positive.txt")
neg_words = load_sentiws("SentiWS_v2.0_Negative.txt")
# Zusammenführen in ein Dictionary
senti_dict = {**pos_words, **neg_words}

def sentiment_score(text):
    tokens = tokenize_and_filter(text)
    score = 0
    for token in tokens:
        if token in senti_dict:
            score += senti_dict[token]
    return score

def tokenize_and_filter(text):
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in stop_words]
