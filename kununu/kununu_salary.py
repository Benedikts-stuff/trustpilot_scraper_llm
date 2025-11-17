from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import random
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd

# Setup
opts = webdriver.ChromeOptions()
#opts.add_argument("--headless")  # Nur aktivieren, wenn du nicht mitverfolgen willst
opts.add_argument("--lang=de-DE")
opts.add_argument("user-agent=Mozilla/5.0 ...")

driver = webdriver.Chrome(options=opts)
driver.get("https://www.kununu.com/de/phoenix-contact/gehalt")
time.sleep(2)  # Warte auf erste DOM-Initialisierung

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
      console.log('✅ CMP-Wrapper und Overlay entfernt.');
      clearInterval(interval);
    }
  }
}, 500);
    """)
    print("✅ Alle bekannten Consent-Elemente ausgeblendet.")
except Exception as e:
    print(f"❌ Fehler beim Banner-Ausblenden: {e}")

# Danach: normal weitermachen mit Scrollen & Button-Klick
# Beispiel:
wait = WebDriverWait(driver, 10)

try:
    while True:
        # Scroll ans Ende der Seite
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Warten, bis DOM evtl. neu geladen wurde

        # Alle Buttons mit dem gesuchten Text erfassen
        buttons = driver.find_elements(By.XPATH, "//button[.//span[text()='MEHR JOBTITEL ANZEIGEN']]")

        if not buttons:
            print("✅ Kein Button mehr sichtbar – fertig.")
            break

        button = buttons[0]

        # Scroll Button ins Sichtfeld – sehr wichtig!
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
        time.sleep(1.5)  # Scroll-Animation abwarten

        try:
            wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='MEHR JOBTITEL ANZEIGEN']]"))).click()
            print("➡️ Button geklickt.")
            time.sleep(random.randint(4, 6))  # Zufällige Wartezeit vor erneutem Scroll
        except Exception as e:
            print(f"⚠️ Button war da, aber nicht klickbar: {e}")
            break

except Exception as e:
    print(f"❌ Fehler im Ablauf: {e}")


soup = BeautifulSoup(driver.page_source, "html.parser")
with open("debug.html", "w", encoding="utf-8") as f:
    f.write(soup.prettify())
driver.quit()

salaries = []

for card in soup.select("a.index__link__zTGAc"):
    if not card:
        continue

    # Position
    pos_tag = card.select_one("h3.p-base-bold")
    position = pos_tag.get_text(strip=True) if pos_tag else "Unbekannt"

    # Gehalt
    salary_tag = card.select_one(".p-base-regular.text-dark-63")
    salary_raw = salary_tag.get_text(strip=True) if salary_tag else ""
    match = re.search(r"([\d\.]+),?(\d{0,2})", salary_raw)
    salary = None
    if match:
        number_str = match.group(1).replace('.', '')
        decimal_part = match.group(2) if match.group(2) else "00"
        salary = float(f"{number_str}.{decimal_part}")

    # Gehaltsangaben / Sample Size
    count_tag = card.select_one(".p-tiny-regular.text-dark-63")
    count_text = count_tag.get_text(strip=True) if count_tag else ""
    match = re.search(r"\d+", count_text)
    sample_size = int(match.group(0)) if match else 0

    print(f"{position}: {salary} € ({sample_size} Angaben)")

    salaries.append({
        "Position": position,
        "Salary": salary,
        "Gehaltsangaben": sample_size
    })

df = pd.DataFrame(salaries)
df.to_excel("kununu_salaries.xlsx", index=False)
