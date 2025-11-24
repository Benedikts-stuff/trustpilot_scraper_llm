# scraper.py
import requests
from bs4 import BeautifulSoup
import json
import time
import pandas as pd
import re
import random

# --- NEUE IMPORTE FÜR SELENIUM (KUNUNU) ---
from selenium import webdriver
import platform
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

import platform # Um Windows/Mac zu erkennen
import os       # Um absolute Pfade zu bauen


# -------------------------------------------

# ==============================================================
# 1. TRUSTPILOT SCRAPER (mit max_pages Limit)
# ==============================================================

def get_reviews_from_page(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        req.raise_for_status()
        delay = random.randint(3, 7)
        time.sleep(delay)
        soup = BeautifulSoup(req.text, 'html.parser')
        reviews_raw = soup.find("script", id="__NEXT_DATA__").string
        reviews_raw = json.loads(reviews_raw)
        return reviews_raw["props"]["pageProps"]["reviews"]
    except (requests.RequestException, json.JSONDecodeError, AttributeError) as e:
        print(f"Fehler beim Holen der Reviews: {e}")
        return []


def scrape_trustpilot_reviews(base_url: str, max_pages: int = 1000):
    """
    Scrapt Trustpilot-Reviews.
    Stoppt, wenn keine Reviews mehr gefunden werden ODER max_pages erreicht ist.
    """
    reviews_data = []
    page_number = 1

    while True:
        # HIER IST DEIN PUNKT 1:
        if page_number > max_pages:
            print(f"Scraping-Limit von {max_pages} Seiten erreicht. Stoppe.")
            break

        print(f"Scrape Trustpilot Seite: {page_number}")

        if page_number == 1:
            url = base_url
        elif '?' in base_url:
            url = f"{base_url}&page={page_number}"
        else:
            url = f"{base_url}?page={page_number}"

        reviews = get_reviews_from_page(url)
        if not reviews:
            print("Keine weiteren Reviews gefunden. Stoppe.")
            break

        for review in reviews:
            data = {
                'Date': pd.to_datetime(review["dates"]["publishedDate"]).strftime("%Y-%m-%d"),
                'Author': review["consumer"]["displayName"],
                'Body': review["text"],
                'Heading': review["title"],
                'Rating': review["rating"],
                'Location': review["consumer"]["countryCode"]
            }
            reviews_data.append(data)

        page_number += 1

    reviews_data = [dict(t) for t in {tuple(d.items()) for d in reviews_data}]
    return reviews_data


# ==============================================================
# 2. KUNUNU SCRAPER (BASIEREND AUF DEINEN SKRIPTEN)
# ==============================================================

def _setup_selenium_driver():
    """
    Initialisiert einen Selenium-Driver von einer LOKALEN chromedriver-Datei.
    Diese Funktion nutzt absolute Pfade und ist CWD-unabhängig.
    """
    print("Initialisiere Selenium WebDriver...")
    opts = webdriver.ChromeOptions()
    # opts.add_argument("--headless")
    opts.add_argument("--lang=de-DE")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # --- ROBUSTE PFAD-LOGIK ---
    try:
        # Finde den absoluten Pfad zu dem Ordner, in dem DIESES Skript (scraper.py) liegt
        script_dir = os.path.dirname(os.path.abspath(__file__))

        local_driver_name = "chromedriver"
        if platform.system() == "Windows":
            local_driver_name = "chromedriver.exe"

        # Baue den absoluten Pfad zur Treiber-Datei (z.B. /Users/benedikt/Projekt/chromedriver)
        local_driver_path = os.path.join(script_dir, local_driver_name)

        print(f"Suche nach Treiber unter absolutem Pfad: {local_driver_path}")

        if not os.path.exists(local_driver_path):
            print("--- FATALER FEHLER ---")
            print(f"Datei {local_driver_name} NICHT unter {local_driver_path} gefunden.")
            print("Bitte stelle sicher, dass 'chromedriver' (Mac) oder 'chromedriver.exe' (Win)")
            print("exakt im selben Ordner wie 'scraper.py' liegt.")
            print("------------------------")
            return None

        # Verwende den absoluten Pfad
        service = Service(executable_path=local_driver_path)
        driver = webdriver.Chrome(service=service, options=opts)
        print(f"WebDriver initialisiert (von lokaler Datei: {local_driver_path}).")
        return driver

    except Exception as e:
        print(f"FEHLER: Konnte WebDriver nicht von '{local_driver_path}' starten. {e}")
        print("Mögliche Gründe:")
        print("1. Die 'chromedriver'-Datei ist beschädigt.")
        print("2. Die 'chromedriver'-Version passt NICHT zu deiner installierten Chrome-Browser-Version.")
        print("3. (Nur Mac) Du hast die 'xattr'- und 'chmod'-Befehle nicht ausgeführt.")
        return None


def _hide_consent_banner(driver):
    """Versucht, das Cookie-Banner zu entfernen (aus kununu_salary.py)."""
    try:
        driver.execute_script("""
        const interval = setInterval(() => {
          const shadowHost = document.querySelector('#usercentrics-cmp-ui');
          if (shadowHost && shadowHost.shadowRoot) {
            const shadowRoot = shadowHost.shadowRoot;
            const cmpWrapper = shadowRoot.querySelector('.cmp-wrapper');
            const overlay = shadowRoot.querySelector('#uc-overlay');
            if (cmpWrapper || overlay) {
              if (cmpWrapper) cmpWrapper.remove();
              if (overlay) overlay.remove();
              console.log('Consent-Banner entfernt.');
              clearInterval(interval);
            }
          }
        }, 500);
        """)
    except Exception:
        pass  # Ignoriere Fehler hier


# --- Kununu Kommentare (Logik aus test.py) ---

def _expand_all_reviews(driver, max_pages_to_click: int):
    """Klickt 'Mehr anzeigen' für Review-Listen (DEIN PUNKT 1)."""
    clicks = 0
    while clicks < max_pages_to_click:
        try:
            # Warte, bis der Button da UND klickbar ist
            wait = WebDriverWait(driver, 10)
            btn = wait.until(EC.element_to_be_clickable((By.ID, "reviews-read-more-cta")))

            if not btn.is_displayed():
                break  # Button ist da, aber unsichtbar

            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.5)
            btn.click()
            print(f"Kununu 'Mehr anzeigen' geklickt ({clicks + 1}/{max_pages_to_click})")

            # Warte, bis der *alte* Button verschwunden/verändert ist
            wait.until(EC.staleness_of(btn))
            time.sleep(random.randint(2, 4))
            clicks += 1

        except (TimeoutException, StaleElementReferenceException):
            print("Kein 'Mehr anzeigen'-Button mehr gefunden oder Timeout. Fertig.")
            break
        except Exception as e:
            print(f"Fehler beim Klicken: {e}. Breche ab.")
            break


def _parse_review_cards(driver):
    """Parst alle geladenen Review-Karten (Logik aus test.py)."""
    reviews = []
    cards = driver.find_elements(By.CSS_SELECTOR, "div.index__reviewBlock__I8pdb")
    print(f"{len(cards)} Review-Karten im DOM gefunden. Parse...")

    for card in cards:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", card)
        time.sleep(0.1)  # Warten auf Lazy-Loading

        # "Alle anzeigen" *innerhalb* einer Karte klicken
        try:
            show_all = card.find_element(By.XPATH, ".//button[normalize-space()='Alle anzeigen']")
            if show_all.is_displayed():
                driver.execute_script("arguments[0].click();", show_all)
                time.sleep(0.1)
        except Exception:
            pass  # Kein "Alle anzeigen"-Button

        try:
            dt = card.find_element(By.CSS_SELECTOR, ".p-tiny-regular.text-dark-63").get_attribute("datetime") or ""
            total_txt = card.find_element(By.CSS_SELECTOR, ".index__score__BktQY").text.strip().replace(",", ".")
            total_rating = float(total_txt)
        except Exception:
            dt, total_rating = "", None

        pros = cons = suggestions = ""
        categories = {}

        factor_blocks = card.find_elements(By.XPATH,
                                           ".//div[contains(@class,'factor') and contains(@class,'p-base-regular')]")
        for fb in factor_blocks:
            try:
                title = fb.find_element(By.CSS_SELECTOR, "h4").text.strip().lower()
                text = fb.find_element(By.CSS_SELECTOR, "p").text.strip()
            except Exception:
                continue

            if "gut am arbeitgeber" in title:
                pros = text
            elif "schlecht am arbeitgeber" in title:
                cons = text
            elif "verbesserungsvorsch" in title:
                suggestions = text
            else:
                try:
                    star_raw = fb.find_element(By.XPATH, ".//*[self::span or self::*][@data-score]").get_attribute(
                        "data-score")
                    score = float(star_raw.replace(",", "."))
                    title_orig = fb.find_element(By.CSS_SELECTOR, "h4").text.strip()  # Originaltitel holen
                    categories[title_orig] = {"score": score, "text": text}
                except Exception:
                    pass  # Kein Bewertungs-Faktor

        reviews.append({
            "Date": dt,
            "Rating": total_rating,
            "Pros": pros,
            "Cons": cons,
            "Suggestions": suggestions,
            "Categories": categories
        })
    return reviews


def scrape_kununu_comments(url: str, max_pages_to_click: int = 50):
    """Hauptfunktion zum Scrapen von Kununu-Kommentaren."""
    print("Starte Kununu-Kommentar-Scraper...")
    driver = _setup_selenium_driver()
    if driver is None: return []

    driver.get(url)
    time.sleep(5)
    _hide_consent_banner(driver)
    time.sleep(2)

    _expand_all_reviews(driver, max_pages_to_click)
    reviews = _parse_review_cards(driver)

    driver.quit()
    print(f"Kununu-Scraping beendet. {len(reviews)} Reviews gefunden.")
    return reviews


# --- Kununu Gehälter (Logik aus kununu_salary.py) ---

def scrape_kununu_salary(url: str, max_pages_to_click: int = 50):
    """Hauptfunktion zum Scrapen von Kununu-Gehältern."""
    print("Starte Kununu-Gehalts-Scraper...")
    driver = _setup_selenium_driver()
    if driver is None: return []

    driver.get(url)
    time.sleep(5)
    _hide_consent_banner(driver)
    time.sleep(2)

    wait = WebDriverWait(driver, 10)
    clicks = 0
    try:
        while clicks < max_pages_to_click:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            try:
                button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='MEHR JOBTITEL ANZEIGEN']]")))
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
                time.sleep(1.5)
                button.click()
                print(f"➡️ Gehalts-Button geklickt ({clicks + 1}/{max_pages_to_click})")
                time.sleep(random.randint(4, 6))
                clicks += 1
            except Exception:
                print("✅ Kein Gehalts-Button mehr sichtbar – fertig.")
                break
    except Exception as e:
        print(f"❌ Fehler im Gehalts-Ablauf: {e}")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    salaries = []
    for card in soup.select("a.index__link__zTGAc"):
        pos_tag = card.select_one("h3.p-base-bold")
        position = pos_tag.get_text(strip=True) if pos_tag else "Unbekannt"

        salary_tag = card.select_one(".p-base-regular.text-dark-63")
        salary_raw = salary_tag.get_text(strip=True) if salary_tag else ""
        match = re.search(r"([\d\.]+),?(\d{0,2})", salary_raw)
        salary = None
        if match:
            number_str = match.group(1).replace('.', '')
            salary = float(f"{number_str}.{match.group(2) or '00'}")

        count_tag = card.select_one(".p-tiny-regular.text-dark-63")
        count_text = count_tag.get_text(strip=True) if count_tag else ""
        match = re.search(r"\d+", count_text)
        sample_size = int(match.group(0)) if match else 0

        salaries.append({
            "Position": position,
            "Salary": salary,
            "Gehaltsangaben": sample_size
        })

    print(f"Kununu-Gehälter-Scraping beendet. {len(salaries)} Positionen gefunden.")
    return salaries