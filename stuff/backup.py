from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# === 1. Setup Chrome (Headless optional) ===
options = Options()
# options.add_argument("--headless")  # Deaktivieren, um zu sehen was passiert
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

# === 2. Kununu-Seite laden ===
url = "https://www.kununu.com/de/simon-kucher/kommentare"  # <-- anpassen
driver.get(url)

# === 3. Automatisches Scrollen bis Seitenende ===
def slow_scroll(driver, step_size=300, pause=0.5, max_pause=3):
    """Scrollt langsam und schrittweise wie ein Mensch"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    current_position = 0

    while current_position < last_height:
        driver.execute_script(f"window.scrollTo(0, {current_position});")
        time.sleep(pause)
        current_position += step_size
        last_height = driver.execute_script("return document.body.scrollHeight")

        # Begrenzung: Warte zusätzlich, wenn wirklich viel geladen wird
        time.sleep(min(pause * 1.5, max_pause))

    # Einmal ganz nach unten am Ende
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

time.sleep(10)
slow_scroll(driver)

# === 4. Warten, bis data-score sichtbar ist (sicher ist sicher) ===
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-score]"))
)

# === 5. Alle data-score-Elemente extrahieren ===
score_elements = driver.find_elements(By.CSS_SELECTOR, "[data-score]")

for i, el in enumerate(score_elements):
    score = el.get_attribute("data-score")
    print(f"[{i+1}] Score: {score}")

driver.quit()
