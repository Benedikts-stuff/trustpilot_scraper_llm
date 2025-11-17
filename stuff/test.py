import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from bs4 import BeautifulSoup
import ast
import time
import random
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Setup
opts = webdriver.ChromeOptions()
# opts.add_argument("--headless")  # Nur aktivieren, wenn du keinen sichtbaren Browser brauchst
opts.add_argument("--lang=de-DE")
opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=opts)
driver.get("https://www.kununu.com/de/herma/kommentare")
time.sleep(random.randint(7, 12))  # Initiales Laden abwarten

def expand_all_reviews(driver, timeout_per_click=10, max_clicks=50):
    def count_cards():
        return len(driver.find_elements(By.CSS_SELECTOR, "div.index__reviewBlock__I8pdb"))

    clicks = 0
    while clicks < max_clicks:
        # Button ggf. (wieder) suchen
        btns = driver.find_elements(By.ID, "reviews-read-more-cta")
        if not btns:
            break
        btn = btns[0]
        if not btn.is_displayed():
            break

        before = count_cards()
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        try:
            btn.click()
        except Exception:
            driver.execute_script("arguments[0].click();", btn)

        # Warten bis EINES der Ereignisse eintritt:
        #  - Card-Zahl steigt
        #  - Button verschwindet / stale wird
        try:
            WebDriverWait(driver, timeout_per_click).until(
                lambda d: count_cards() > before or not btn.is_displayed()
            )
        except Exception:
            # letzter Versuch: auf Staleness warten
            try:
                WebDriverWait(driver, 2).until(EC.staleness_of(btn))
            except Exception:
                # nichts hat sich verändert → Abbruch
                break

        clicks += 1

def parse_all_cards(driver):
    reviews = []
    cards = driver.find_elements(By.CSS_SELECTOR, "div.index__reviewBlock__I8pdb")

    for card in cards:
        # in den Viewport, damit IntersectionObserver triggert
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", card)
        time.sleep(0.15)

        # Falls die Card einen "Alle anzeigen"-Button hat, einmal klicken
        try:
            show_all = card.find_element(By.XPATH, ".//button[normalize-space()='Alle anzeigen']")
            if show_all.is_displayed():
                try:
                    show_all.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", show_all)
                # ganz kurz warten, bis neue Faktoren im DOM sind
                time.sleep(0.15)
        except Exception:
            pass

        # Datum
        try:
            time_el = card.find_element(By.CSS_SELECTOR, ".p-tiny-regular.text-dark-63")
            dt = time_el.get_attribute("datetime") or ""
        except Exception:
            dt = ""

        # Gesamt-Rating (Text z.B. "4,8")
        try:
            total_txt = card.find_element(By.CSS_SELECTOR, ".index__score__BktQY").text.strip().replace(",", ".")
            total_rating = float(total_txt)
        except Exception:
            total_rating = None

        # Rolle
        role = ""
        try:
            info = card.find_element(By.CSS_SELECTOR, ".index__sentence__j5Cc3.text-dark-63.index__middot__jwHNi").text.strip()
            m = re.search(r"im Bereich (.+?) bei", info)
            role = m.group(1).strip() if m else ""
        except Exception:
            pass

        # Faktoren
        pros = cons = suggestions = ""
        categories = {}

        factor_blocks = card.find_elements(By.XPATH, ".//div[contains(@class,'factor') and contains(@class,'p-base-regular')]")
        for fb in factor_blocks:
            # jeden Faktor in den Viewport holen (triggert star-render)
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", fb)
            time.sleep(0.05)

            # Titel & Text
            try:
                title = fb.find_element(By.CSS_SELECTOR, "h4").text.strip()
            except Exception:
                continue
            try:
                text = fb.find_element(By.CSS_SELECTOR, "p").text.strip()
            except Exception:
                text = ""

            tl = title.lower()
            if "gut am arbeitgeber" in tl or "pro" in tl:
                pros = text
                continue
            if "schlecht am arbeitgeber" in tl or "contra" in tl:
                cons = text
                continue
            if "verbesserungsvorsch" in tl:
                suggestions = text
                continue

            # Nur echte Bewertungs-Faktoren mit data-score einsammeln
            stars = fb.find_elements(By.XPATH, ".//div[contains(@class,'scoreBlock')]//span[@data-score]")
            if not stars:
                # zweiter Versuch: irgend ein Descendant mit data-score (Hash-Fallback)
                stars = fb.find_elements(By.XPATH, ".//*[self::span or self::*][@data-score]")
            if not stars:
                continue

            raw = (stars[0].get_attribute("data-score") or "").strip()
            try:
                score = float(raw.replace(",", "."))
            except Exception:
                score = None

            categories[title] = {"score": score, "text": text}

        reviews.append({
            "Date": dt,
            "Rating": total_rating,
            "Role": role,
            "Pros": pros,
            "Cons": cons,
            "Suggestions": suggestions,
            "Categories": categories
        })
    return reviews

expand_all_reviews(driver)              # <- paginiert stabil durch
reviews = parse_all_cards(driver)       # <- sammelt sauber je Card
# jetzt erst den Driver schließen
driver.quit()

# Speichern wie gehabt
df = pd.DataFrame(reviews)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
CATEGORY_COLUMNS = [
    "Gehalt/Sozialleistungen",
    "Image",
    "Karriere/Weiterbildung",
    "Arbeitsatmosphäre",
    "Kommunikation",
    "Kollegenzusammenhalt",
    "Work-Life-Balance",
    "Vorgesetztenverhalten",
    "Interessante Aufgaben",
    "Arbeitsbedingungen",
    "Umwelt-/Sozialbewusstsein",
    "Gleichberechtigung",
    "Umgang mit älteren Kollegen",
]

def _as_dict(x):
    # falls 'Categories' schon ein dict ist → zurückgeben
    if isinstance(x, dict):
        return x
    # falls es als String im DF liegt (kommt vor) → sicher parsen
    if isinstance(x, str) and x.strip():
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return {}

def _get_score(catdict, name):
    try:
        return catdict.get(name, {}).get("score", None)
    except Exception:
        return None

# Für jede Kategorie eine Spalte mit dem Score (oder NaN)
cat_series = df["Categories"].apply(_as_dict)
for col in CATEGORY_COLUMNS:
    df[col] = cat_series.apply(lambda d: _get_score(d, col))

df.to_excel("kununu_Herma_comments.xlsx", index=False)
print("Fertig: kununu_Herma_comments.xlsx")
