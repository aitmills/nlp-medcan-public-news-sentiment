# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:14:57 2023

@author: aitmi
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import PyPDF2
from io import BytesIO


def get_search_results(query, site, api_key, num_results=10, page=1):
    
    start = (page - 1) * num_results + 1
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={site}&q={query}&num={num_results}&start={start}&excludeTerms=/topics/"
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, timeout=10)  # Added a timeout of 10 seconds
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to fetch search results. Error details: {e}")
        return {"error": str(e)}

    search_results = response.json()
    

    for result in search_results.get('items', []):

        pagemap = result.get('pagemap', {})
        cse_thumbnail = pagemap.get('cse_thumbnail', [{}])[0]
        metatags = pagemap.get('metatags', [{}])[0]

        date_published = cse_thumbnail.get('datepublished') or metatags.get('datepublished') or metatags.get('datecreated')
        result['date_from_search'] = date_published if date_published else ""
        result['snippet'] = result.get('snippet', '')  # Store the snippet in the result dictionary

    return search_results

def scrape_article(url):
    # Check if the URL is a PDF file
    if url.lower().endswith(".pdf"):
        return "", "", extract_text_from_pdf(url)
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to fetch URL {url}. Error details: {e}")
        return "", "", ""
    
    soup = BeautifulSoup(response.content, "html.parser")
    # Start by searching for an <h1> element to get the headline
    headline = soup.find("h1")

    # If the <h1> element contains "Now Playing", search for other alternatives
    if headline and "Now Playing" in headline.text:
        headline = soup.find("h2")

    headline = headline.text if headline else ""

    # Extract the date
    date = soup.find("time")
    date = date.text if date else ""

    # Extract the article body
    # Extract the article body
    try:
        # List of common class names for the article content
        common_class_names = [
            "content",
            "body",
            "bodyColumn"
            "article__main-content articleContent"
            "article__body-croppable",
            "new-article-body",
            "entry-content",
            "post-content",
            "article-content",
            "article-body",
            "field-name-body",
            "col-md-11 fix-for-content-col",
            "col-md-12 content-main clinical-services hospital_and_healthcare"
        ]

        article_content = None

        # Iterate through the common class names and find the first match
        for class_name in common_class_names:
            article_content = soup.find("div", class_=class_name)
            if article_content:
                break

        if article_content:
            body = " ".join([p.text for p in article_content.find_all("p")])
        else:
            body = ""
    except:
        body = ""

    return headline, date, body

def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    pdf_data = BytesIO(response.content)
    pdf_reader = PyPDF2.PdfFileReader(pdf_data)

    num_pages = pdf_reader.numPages
    text = []

    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        text.append(page.extractText())

    return " ".join(text)

def main(domain='theconversation.com'):
    print("Searching for Australian news articles related to medicinal cannabis...")
    query = f"medicinal cannabis site:{domain}"
    site = "441cd65b2c44d445b"
    api_key = "AIzaSyAHjM4--WZZCqINCUgwbvSoM-K1iwPLpEA"
    num_results = 10
    num_pages = 10  # Change this value to control the number of pages to scrape

    articles = []

    for page in range(1, num_pages + 1):
        print(f"Retrieving page {page} search results...")
        search_results = get_search_results(query, site, api_key, num_results, page)

        if 'items' not in search_results:
            print(f"Error: Unable to retrieve page {page} search results.")
            if 'error' in search_results:
                print(search_results['error']['message'])
            continue

        for result in search_results['items']:
            url = result['link']
            print(f"Scraping {url}...")

            # Fetch the list of article URLs on the search result page
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                article_links = soup.find_all("a", class_="article-link")
            except requests.exceptions.RequestException as e:
                print(f"Error: Unable to fetch URL {url}. Error details: {e}")
                continue

            # Scrape each article URL
            for article_link in article_links:
                article_url = f"https://{domain}{article_link['href']}"
                print(f"Scraping {article_url}...")
                snippet = result['snippet']  # Get the snippet from the result dictionary
                headline, date_from_article, body = scrape_article(article_url)
                date = result['date_from_search'] if result['date_from_search'] else date_from_article
                articles.append({"url": article_url, "headline": headline or "", "date": date or "", "body": body or "", "snippet": snippet or ""})

    df = pd.DataFrame(articles)
    df['headline_word_count'] = df.get('headline', '').apply(lambda x: len(x.split()))
    df['body_word_count'] = df['body'].apply(lambda x: len(x.split()))

    print("Data stored in dataframe with word counts:")
    print(df.head())

    return df


if __name__ == "__main__":
    main()
