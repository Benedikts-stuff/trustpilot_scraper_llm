# trustpilot_scraper/scraper.py
import random

import pandas
import requests
from bs4 import BeautifulSoup
import json
import time
import pandas as pd
import re

from openpyxl.styles.builtins import total


def get_reviews_from_page(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        req.raise_for_status()  # Raise an error for bad status codes
        delay = random.randint(7,12)
        time.sleep(delay)  # Add a delay to avoid overwhelming the server
        soup = BeautifulSoup(req.text, 'html.parser')
        reviews_raw = soup.find("script", id="__NEXT_DATA__").string
        reviews_raw = json.loads(reviews_raw)
        return reviews_raw["props"]["pageProps"]["reviews"]
    except (requests.RequestException, json.JSONDecodeError, AttributeError) as e:
        return []

def get_overall_rating(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'html.parser')

        rating_box = soup.select_one('.styles_ratingDistributionCard__qgoBg')

        trust_score_raw = rating_box.select_one(
            '.CDS_Typography_appearance-default__bedfe1.CDS_Typography_display-l__bedfe1.styles_trustScore__MVJJI'
        ).text.strip()
        trust_score = float(trust_score_raw.replace(",", "."))

        rating_title = rating_box.select_one(
            '.CDS_Typography_appearance-default__bedfe1.CDS_Typography_heading-xxs__bedfe1.styles_starRatingName__njtqK'
        ).text.strip()

        number_ratings = rating_box.select_one(
            '.CDS_Typography_appearance-default__bedfe1.CDS_Typography_body-s__bedfe1.styles_reviewCount__NXlel'
        ).text.strip()
        total_ratings = int(re.sub(r"[^\d]", "", number_ratings))

        # Sterne-Verteilung
        stars = {}
        rows = rating_box.select('.rating-distribution-row_row__TH3OE')  # Punkt wichtig!
        for row in rows:
            star_label = row.get("data-star-rating")
            style = row.select_one("span.rating-distribution-row_barValue__iFje4")["style"]
            percent = float(re.search(r"width:([\d.]+)%", style).group(1))
            stars[star_label] = percent

        # Absolutwerte berechnen
        stars_absolute = {k: round(total_ratings * v / 100) for k, v in stars.items()}

        return trust_score, rating_title, total_ratings, stars, stars_absolute

    except (requests.RequestException, AttributeError, ValueError) as e:
        print(f"Fehler: {e}")
        return None


def scrape_trustpilot_reviews(base_url: str):
    reviews_data = []

    page_number = 1
    while True:
        print("PageNumber: ", page_number)
        #url = f"{base_url}?page={page_number}"
        url = f"{base_url}"
        if page_number>1:
            url = f"{base_url}&page={page_number}"

        reviews = get_reviews_from_page(url)

        if not reviews:
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

    # Remove duplicates based on the 'Body' field
    reviews_data = [dict(t) for t in {tuple(d.items()) for d in reviews_data}]
    
    return reviews_data