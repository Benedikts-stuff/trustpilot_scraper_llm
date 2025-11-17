import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from bs4 import BeautifulSoup
import time
import random

# 1. Setup
opts = webdriver.ChromeOptions()
# opts.add_argument("--headless")  # Nur aktivieren, wenn du keinen sichtbaren Browser brauchst
opts.add_argument("--lang=de-DE")
opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=opts)
driver.get("https://www.kununu.com/de/herma/kommentare?date=6months")
time.sleep(random.randint(7, 12))  # Initiales Laden abwarten

# 2. Alle "Mehr anzeigen"-Buttons klicken
while True:
    try:
        button = driver.find_element(By.ID, "reviews-read-more-cta")
        if button.is_displayed():
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                time.sleep(0.5)
                button.click()
                print("Mehr anzeigen geklickt")
                time.sleep(random.randint(6, 10))  # Warten auf Nachladen
            except ElementClickInterceptedException:
                print("Button wurde gefunden, aber konnte nicht geklickt werden (überdeckt?) – versuche JS-Klick")
                driver.execute_script("arguments[0].click();", button)
                time.sleep(random.randint(6, 10))
        else:
            print("Button ist nicht mehr sichtbar – fertig.")
            break
    except NoSuchElementException:
        print("Kein 'Mehr anzeigen'-Button gefunden – fertig.")
        break

# 3. Optional: sanft nach unten scrollen, damit alle Inhalte geladen sind
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# 4. HTML speichern und analysieren
soup = BeautifulSoup(driver.page_source, "html.parser")
with open("debug.html", "w", encoding="utf-8") as f:
    f.write(soup.prettify())

print("HTML gespeichert – bereit zum Parsen.")
driver.quit()
# 4. Daten extrahieren
reviews = []
#review_cards = driver.find_elements(By.CSS_SELECTOR, "div.index__reviewBlock__I8pdb")

for card in soup.select("div.index__reviewBlock__I8pdb"):
    review_area = card
    #print("-----BLOCK------")
    #print(review_area.prettify())

    if not review_area:
        continue

    time_tag = review_area.select_one(".p-tiny-regular.text-dark-63")
    datetime = time_tag["datetime"] if time_tag else ""

    stars_text = card.select_one(".index__score__BktQY").text.strip().replace(",", ".")
    stars = float(stars_text)

    elem = card.select_one(".index__sentence__j5Cc3.text-dark-63.index__middot__jwHNi")
    match = None
    if elem:
        match = re.search(r"im Bereich (.+?) bei", elem.get_text(strip=True))

    if match:
        role = match.group(1).strip()
    else:
        role = ""

    positive = ""
    negative = ""
    suggestions = ""
    for block in review_area.select("div.index__factor__Mo6xW.p-base-regular"):
        title_tag = block.select_one("h4")
        content_tag = block.select_one("p")

        if not title_tag or not content_tag:
            continue

        title = title_tag.text.strip()
        content = content_tag.get_text(separator="\n").strip()

        if "Gut am Arbeitgeber finde ich" in title:
            positive = content
        elif "Schlecht am Arbeitgeber finde ich" in title:
            negative = content
        elif "Verbesserungsvorschläge" in title:
            suggestions = content

    categories = {}

    for block in review_area.select("div.index__factor__Mo6xW.p-base-regular"):
        title_tag = block.select_one("h4")
        star_tag = block.select_one("span[data-score]")
        text_tag = block.select_one("p")

        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        category_score = float(star_tag["data-score"].replace(",", ".")) if star_tag else None
        text = text_tag.get_text(strip=True) if text_tag else ""

        print(f"Category: {title}, Score: {category_score}, Text: {text}")
        categories[title] = {
            "score": category_score,
            "text": text
        }

    reviews.append({
        "Date": datetime,
        "Rating": stars,
        "Role": role,
        "Pros": positive,
        "Cons":  negative,
        "Suggestions": suggestions,
        "Categories": categories
    })

print(reviews)
# 5. Datenframe & Speichern
df = pd.DataFrame(reviews)
df["Date"] = pd.to_datetime(df["Date"]).dt.date
df.to_excel("kununu_Herma_comments.xlsx", index=False)
