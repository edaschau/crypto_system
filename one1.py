import sqlite3
import time
from datetime import datetime, timezone, timedelta
import json
from dateutil import parser
import logging
import random
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from together import Together
from pydantic import BaseModel, Field
import csv
import requests
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
DB_NAME = 'crypto.db'

# API configuration
API_KEY = "e0bacd011dfb18c6f919df4f9157dfe0b79037c9f7f03f55dbcbf397e3d4e140"
client = Together(api_key=API_KEY)

# Models configuration
MODELS = [
    {
        "name": "google/gemma-2-9b-it",
        "params": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 0.5,
            "top_k": 40,
        }
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "params": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 0.5,
            "top_k": 40,
        }
    },
]

# Crypto keywords
CRYPTO_KEYWORDS = {
    'Bitcoin': ['bitcoin', 'btc'],
    'Ethereum': ['ethereum', 'eth'],
    'Dogecoin': ['dogecoin', 'doge']
}

# Template data
template_data = {
    'name': 'yahoo_finance',
    'template': {
        'title': 'h1[data-test-locator="headline"]',
        'author': 'span.caas-author-byline-collapse',
        'datetime': {'selector': 'time', 'attribute': 'datetime', 'index': [0]},
        'article': 'div.caas-body',
        'ticker_symbols': {'selector': 'div.caas-body-content', 'attribute': 'data-symbol', 'index': [0], 'inner': {'selector': 'fin-ticker', 'attribute': 'symbol'}},
        'source': 'a[class="link caas-attr-provider-logo"]',
        'sourcr_url': {'selector': 'a[class="link caas-attr-provider-logo"]', 'attribute': 'href', 'index': [0]},
    }
}

templates = {template_data['name']: template_data['template']}

# Twitter API configuration
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAL94uQEAAAAA6mm4EjCpZ%2B6Pl6hBsvitjmJ7UFM%3DvDbsFbnbmqAaVDTivROUtgCG3p8DzZxi1SppiMz9uOYwvTwA9L"
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Twitter usernames for each cryptocurrency
TWITTER_USERNAMES = {
    'Bitcoin': ['elonmusk', 'saylor', 'cz_binance', 'jack', 'coinbase'],
    'Ethereum': ['binance', 'coinbase', 'ethereum', 'VitalikButerin', 'arbitrum'],
    'Dogecoin': ['elonmusk', 'MattWallace888', 'mcuban', 'SnoopDogg', 'coinbase']
}

def load_prompts(file_name):
    with open(file_name, 'r') as f:
        reader = csv.DictReader(f)
        return [{k.lower().replace(' ', '_'): v for k, v in row.items()} for row in reader]

NEWS_PROMPTS = load_prompts('news_prompts.csv')
TWITTER_PROMPTS = load_prompts('twitter_prompts.csv')

def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    # Create links and articles tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS links (
            url TEXT PRIMARY KEY,
            first_seen_utc TIMESTAMP,
            first_seen_unix INTEGER,
            is_scraped INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            datetime_utc TIMESTAMP,
            datetime_unix INTEGER,
            content TEXT,
            ticker_symbols TEXT,
            FOREIGN KEY (url) REFERENCES links (url)
        )
    """)
    
    # Create tables for each crypto type and model (for news)
    for crypto in CRYPTO_KEYWORDS.keys():
        for model in MODELS:
            table_name = f"{crypto}_{model['name'].replace('/', '_').replace('-', '_').replace('.', '_')}_news"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    url TEXT PRIMARY KEY,
                    FOREIGN KEY (url) REFERENCES articles (url)
                )
            """)
            
            # Add columns for each prompt dynamically
            for prompt in NEWS_PROMPTS:
                column_name = prompt['aspect'].lower().replace(' ', '_')
                try:
                    cur.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" INTEGER')
                except sqlite3.OperationalError:
                    pass
    
    # Create Twitter tables for each crypto type and model
    for crypto in CRYPTO_KEYWORDS.keys():
        for model in MODELS:
            table_name = f"{crypto}_{model['name'].replace('/', '_').replace('-', '_').replace('.', '_')}_twitter"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    tweet_id TEXT PRIMARY KEY,
                    author_id TEXT,
                    text TEXT,
                    created_at TIMESTAMP
                )
            """)
            
            # Add columns for each prompt dynamically
            for prompt in TWITTER_PROMPTS:
                column_name = prompt['aspect'].lower().replace(' ', '_')
                try:
                    cur.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" INTEGER')
                except sqlite3.OperationalError:
                    pass
    
    conn.commit()
    conn.close()
def initialize_browser():
    options = Options()
    # options.add_argument('-headless')
    options.set_preference('permissions.default.image', 2)
    options.set_preference('javascript.enabled', False)
    options.set_preference('media.autoplay.default', 5)
    options.set_preference('media.volume_scale', '0.0')
    service = Service('geckodriver')  # Replace with the path to your geckodriver
    return webdriver.Firefox(service=service, options=options)

def scrape_and_store_links(driver):
    url = "https://finance.yahoo.com/topic/crypto/"
    driver.get(url)
    wait = WebDriverWait(driver, 3)
    wait.until(EC.presence_of_element_located((By.ID, "Fin-Stream")))

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    fin_stream = soup.find('div', id='Fin-Stream')
    if fin_stream:
        links = [a['href'] for a in fin_stream.find_all('a', href=True)]
    else:
        logging.warning("Fin-Stream div not found")
        return []

    now = datetime.now(timezone.utc)
    now_unix = int(now.timestamp())
    
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    new_links = []
    for link in links:
        if ("/news/" in link) and (".html" in link) and ("https:" in link):
            cur.execute("""
                INSERT OR IGNORE INTO links (url, first_seen_utc, first_seen_unix)
                VALUES (?, ?, ?)
            """, (link, now, now_unix))
            if cur.rowcount > 0:
                logging.info(f"Added new link: {link}")
                new_links.append(link)
    
    conn.commit()
    conn.close()
    
    logging.info(f"Link scraping completed at {now}")
    return new_links

def extract_article_data(driver, url):
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 3)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        template = templates.get('yahoo_finance')
        if template:
            article_data = {}
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            def extract_elements(selector_info, parent_element):
                selector = selector_info['selector']
                attribute = selector_info['attribute']
                index = selector_info.get('index')
                inner = selector_info.get('inner', [])

                elements = parent_element.select(selector)
                if elements:
                    if index:
                        elements = [elements[i] for i in index if i < len(elements)]
                    value = []
                    for element in elements:              
                        if inner:
                            inner_value = []
                            inner_value.extend(extract_elements(inner, element))
                            value.append(inner_value)
                        else:
                            if attribute == 'text':
                                value.append(element.get_text(strip=True))
                            else:
                                value.append(element.get(attribute))
                    return value
                else:
                    return []
                
            for field, selector_info in template.items():
                try:
                    if isinstance(selector_info, dict):
                        value = extract_elements(selector_info, soup)
                        article_data[field] = value
                    else:
                        element = soup.select_one(selector_info)
                        if element:
                            article_data[field] = element.get_text(strip=True)
                        else:
                            article_data[field] = ''
                except Exception as e:
                    logging.error(f"Error extracting {field}: {e}")
                    article_data[field] = ''

            return article_data
        else:
            logging.warning("Template 'yahoo_finance' not found.")
            return None
    except Exception as e:
        logging.error(f"Unexpected error during extraction for URL {url}: {e}")
        return None

def store_article(url, article_data):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    title = str(article_data.get('title')) if article_data.get('title') is not None else None
    author = str(article_data.get('author')) if article_data.get('author') is not None else None
    datetime_str = str(article_data.get('datetime')[0]) if article_data.get('datetime') is not None else None
    content = str(article_data.get('article')) if article_data.get('article') is not None else None
    ticker_symbols = json.dumps(article_data.get('ticker_symbols')) if article_data.get('ticker_symbols') is not None else None
    
    datetime_utc = None
    datetime_unix = None
    if datetime_str:
        try:
            parsed_date = parser.parse(datetime_str)
            datetime_utc = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            datetime_unix = int(time.mktime(parsed_date.timetuple()))
        except ValueError:
            logging.warning(f"Invalid date format for URL {url}: {datetime_str}")    
    
    cur.execute("""
        INSERT OR REPLACE INTO articles (url, title, author, content, datetime_utc, datetime_unix, ticker_symbols)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (url, title, author, content, datetime_utc, datetime_unix, ticker_symbols))
    
    cur.execute("UPDATE links SET is_scraped = 1 WHERE url = ?", (url,))
    
    conn.commit()
    conn.close()

    return title, content, datetime_utc

def get_unscraped_links():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT url FROM links WHERE is_scraped = 0")
    links = [row[0] for row in cur.fetchall()]
    conn.close()
    return links

def get_crypto_type(title):
    title_lower = title.lower()
    for crypto, keywords in CRYPTO_KEYWORDS.items():
        if any(keyword in title_lower for keyword in keywords):
            return crypto
    return None

def get_model_responses(message, model, crypto_name, is_twitter=False):
    responses = {}
    
    prompts = TWITTER_PROMPTS if is_twitter else NEWS_PROMPTS
    content_type = "tweet" if is_twitter else "news article"
    
    system_prompt = f"""
    You are an expert in analyzing {content_type}s about cryptocurrency and {crypto_name}. 
    You will be given a {content_type} about {crypto_name}. For each aspect below, provide a single integer score from 1 to 10 based on the content.
    Only respond with the scores, one per line, in the order they are listed.
    Do not include any other text or explanations.

    Aspect to rate:
    """
    
    for prompt in prompts:
        system_prompt += f"\n{prompt['prompt']}"
    
    system_prompt += f"\n\nRemember, only output {len(prompts)} integers from 1 to 10, one per line, nothing else."
    
    response = client.chat.completions.create(
        model=model['name'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        **model['params']
    )
    
    try:
        scores = response.choices[0].message.content.strip().split('\n')
        for prompt, score in zip(prompts, scores):
            try:
                score_int = int(score)
                if 1 <= score_int <= 10:
                    responses[prompt['aspect']] = score_int
                else:
                    responses[prompt['aspect']] = 5  # Default to neutral if out of range
            except ValueError:
                logging.warning(f"Invalid response for {prompt['aspect']}: {score}")
                responses[prompt['aspect']] = 5  # Default to neutral if parsing fails
    except Exception as e:
        logging.error(f"Error processing model response: {e}")
        responses = {prompt['aspect']: 5 for prompt in prompts}  # Default all to neutral if processing fails
    
    return responses

def process_article(url, title, content, datetime_utc):
    crypto_type = get_crypto_type(title)
    if crypto_type:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        
        for model in MODELS:
            try:
                sentiment = get_model_responses(f"Title: {title}\n\nContent: {content}", model, crypto_type, is_twitter=False)
                
                # Sanitize the table name
                table_name = f"{crypto_type}_{model['name'].replace('/', '_').replace('-', '_').replace('.', '_')}_news"
                
                # Convert aspect names to lowercase for column names
                columns = ', '.join([f'"{key.lower().replace(" ", "_")}"' for key in sentiment.keys()])
                placeholders = ', '.join(['?' for _ in sentiment])
                values = tuple(sentiment.values())
                
                cur.execute(f"""
                    INSERT OR REPLACE INTO "{table_name}" 
                    (url, {columns})
                    VALUES (?, {placeholders})
                """, (url, *values))
                
                logging.info(f"Processed and stored {crypto_type} article for model {model['name']}: {url}")
            except Exception as e:
                logging.error(f"Error processing article {url} for model {model['name']}: {e}")
        
        conn.commit()
        conn.close()
    else:
        logging.info(f"Article not related to tracked cryptocurrencies: {url}")

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

import time
import requests

def connect_to_endpoint(url, params):
    max_retries = 5
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url, auth=bearer_oauth, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logging.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(response.status_code, response.text)
        except Exception as e:
            logging.error(f"Error connecting to Twitter API: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

    raise Exception("Max retries reached. Unable to connect to Twitter API.")


def get_twitter_data(crypto_name):
    end_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    start_time = end_time - timedelta(hours=1)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    tweets = []
    for username in TWITTER_USERNAMES[crypto_name]:
        query_params = {
            'query': f'from:{username} {crypto_name.lower()}',
            'start_time': start_time_str,
            'end_time': end_time_str,
            'expansions': 'author_id',
            'tweet.fields': 'created_at,text'
        }
        json_response = connect_to_endpoint(SEARCH_URL, query_params)
        if 'data' in json_response:
            tweets.extend(json_response['data'])
    
    return tweets

def process_twitter_data(crypto_name):
    tweets = get_twitter_data(crypto_name)
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    for model in MODELS:
        table_name = f"{crypto_name}_{model['name'].replace('/', '_').replace('-', '_').replace('.', '_')}_twitter"
        
        for tweet in tweets:
            sentiment = get_model_responses(f"Tweet: {tweet['text']}", model, crypto_name, is_twitter=True)
            
            columns = ', '.join([f'"{key.lower().replace(" ", "_")}"' for key in sentiment.keys()])
            placeholders = ', '.join(['?' for _ in sentiment])
            values = tuple(sentiment.values())
            
            cur.execute(f"""
                INSERT OR REPLACE INTO "{table_name}" 
                (tweet_id, author_id, text, created_at, {columns})
                VALUES (?, ?, ?, ?, {placeholders})
            """, (tweet['id'], tweet['author_id'], tweet['text'], tweet['created_at'], *values))
            
            logging.info(f"Processed and stored {crypto_name} tweet for model {model['name']}: {tweet['id']}")
    
    conn.commit()
    conn.close()


def main():
    initialize_database()
    driver = initialize_browser()

    try:
        while True:
            # Scrape and process news
            new_links = scrape_and_store_links(driver)
            unscraped_links = get_unscraped_links()
            
            for link in unscraped_links:
                article_data = extract_article_data(driver, link)
                if article_data:
                    title, content, datetime_utc = store_article(link, article_data)
                    process_article(link, title, content, datetime_utc)
                else:
                    logging.warning(f"Failed to extract data from {link}")
                time.sleep(1)
            
            # Process Twitter data
            for crypto_name in CRYPTO_KEYWORDS.keys():
                process_twitter_data(crypto_name)
            
            # Wait before the next cycle
            time.sleep(5)  # Wait for 1 hour

    except KeyboardInterrupt:
        logging.info("Scraper manually terminated.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()