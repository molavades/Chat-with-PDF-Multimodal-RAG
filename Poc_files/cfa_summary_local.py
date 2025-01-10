import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re

# Configure Selenium (headless mode for faster scraping)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def clean_title(title):
    """Remove non-word characters except spaces, allowing Chinese and alphabetic characters."""
    return re.sub(r'[^\w\s]', '', title, flags=re.UNICODE).strip()

def save_text_file(content, folder, filename):
    """Save content to a local file inside a specific folder."""
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Error creating directory {folder}. Error: {e}")
            return

    # Clean the filename and ensure uniqueness
    filename = clean_title(filename) + ".txt"
    file_path = os.path.join(folder, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved {filename} to {folder}")
    except Exception as e:
        print(f"Failed to save {filename}. Error: {e}")

def get_body_text_from_publication(publication_url):
    """Extract the body text from multiple div classes and standalone p tags, ignoring unwanted elements."""
    try:
        driver.get(publication_url)
        time.sleep(3)  # Give the page some time to load

        publication_soup = BeautifulSoup(driver.page_source, "lxml")
        
        # List of possible div classes to find the body text
        possible_classes = ["paragraph", "article__paragraph", "article-description", "overview__content"]
        body_text = []

        # Extract text from all divs with matching classes, ignoring footer and modal sections
        for class_name in possible_classes:
            body_divs = publication_soup.find_all("div", class_=class_name)
            for body_div in body_divs:
                if body_div.find_parent(class_="footer") or body_div.find_parent(class_="modal-body"):
                    continue  # Skip this div
                
                paragraphs = body_div.find_all("p")  # Find all <p> tags inside the div
                for p in paragraphs:
                    if 'call-out__description' in p.get('class', []):
                        continue
                    body_text.append(p.get_text(strip=True))

        # Also extract text from all standalone <p> tags
        standalone_paragraphs = publication_soup.find_all("p")
        for p in standalone_paragraphs:
            if p.find_parent(class_="footer") or p.find_parent(class_="modal-body"):
                continue  # Skip this paragraph
            if 'call-out__description' in p.get('class', []):
                continue
            if not any(p.find_parent(class_=class_name) for class_name in possible_classes):
                body_text.append(p.get_text(strip=True))
                    
        # Join the collected text into a single string
        if body_text:
            return "\n".join(body_text)
        else:
            print(f"No text found for publication: {publication_url}")
            return None

    except Exception as e:
        print(f"Error accessing {publication_url}: {e}")
        return None

def scrape_page_for_texts(page_url, scraped_titles):
    """Scrape all publications on a specific page and save their body text."""
    try:
        driver.get(page_url)
        time.sleep(3)  # Wait for the page to load

        soup = BeautifulSoup(driver.page_source, "lxml")
        publications = soup.select(".CoveoResultLink")

        if not publications:
            print(f"No publications found at {page_url}")
            return False  # No publications found, stop the loop

        new_publication_found = False  # Track if any new publication is found on the page

        for pub in publications:
            title = pub.text.strip()  # Extract the title
            pub_url = pub["href"] if pub.has_attr("href") else None

            if pub_url and title not in scraped_titles:
                print(f"Title: {title}")
                print(f"Publication URL: {pub_url}")

                # Check if pub_url is complete
                if not pub_url.startswith("http"):
                    print(f"Skipping incomplete URL: {pub_url}")
                    continue  # Skip this publication if the URL is incomplete

                # Visit the publication page to extract the body text
                body_text = get_body_text_from_publication(pub_url)

                # Save body text if found
                if body_text:
                    # Clean the title before saving
                    print(f"Saving body text for: {title}\n")
                    save_text_file(body_text, "cfa_texts", title)
                    scraped_titles.add(title)  # Add the title to avoid duplicates
                    new_publication_found = True
                else:
                    print(f"No body text found for {title}. URL: {pub_url}")

        if not new_publication_found:
            print("No new publications found on this page.")
            return False  # Stop if no new publications were found

        return True  # New publications found, continue to the next page

    except Exception as e:
        print(f"Error scraping page {page_url}: {e}")
        return False

def scrape_all_pages_for_texts():
    """Scrape all pages dynamically until no more results are found."""
    base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
    page_number = 0
    scraped_titles = set()  # Set to keep track of processed titles

    while True:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")
        print(f"Page number: {page_number}\n")

        # Scrape the current page for body text
        if not scrape_page_for_texts(page_url, scraped_titles):
            break  # Stop if no more publications are found or no new publications found

        page_number += 1

# Run the scraper to save body texts
scrape_all_pages_for_texts()

# Quit the WebDriver
driver.quit()
