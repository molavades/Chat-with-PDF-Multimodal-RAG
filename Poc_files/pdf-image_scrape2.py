import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Configure Selenium (headless mode for faster scraping)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def save_file(content, folder, filename):
    """Save content to a local file inside a specific folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Saved {filename} to {folder}")
    except Exception as e:
        print(f"Failed to save {filename}. Error: {e}")

def download_file(url, folder, filename):
    """Download the file from the given URL and save it locally."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        save_file(response.content, folder, filename)
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")

def get_image_and_pdf_from_publication(publication_url, title):
    """Extract the image and PDF link from the publication page."""
    try:
        driver.get(publication_url)
        time.sleep(3)  # Give the page some time to load

        publication_soup = BeautifulSoup(driver.page_source, "lxml")
        
        # Define base URL to handle relative image paths
        base_url = "https://rpc.cfainstitute.org"

        # Extract the PDF link
        pdf_url = None
        links = publication_soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if href.lower().endswith(".pdf"):
                if not href.startswith("http"):
                    href = f"{base_url}{href}"
                pdf_url = href
                break  # Exit after finding the first PDF link

        # Extract the image link by getting the 'src' from the <img> tag
        image_url = None
        image_tag = publication_soup.find("img", class_="article-cover")  # Find the image with the class "article-cover"
        if image_tag and image_tag.has_attr("src"):
            img_src = image_tag["src"]
            if not img_src.startswith("http"):
                img_src = f"{base_url}{img_src}"  # Handle relative URLs
            image_url = img_src

        return pdf_url, image_url

    except Exception as e:
        print(f"Error accessing {publication_url}: {e}")
        return None, None

def scrape_page(page_url):
    """Scrape all publications on a specific page."""
    driver.get(page_url)
    time.sleep(3)  # Wait for the page to load

    soup = BeautifulSoup(driver.page_source, "lxml")
    publications = soup.select(".CoveoResultLink")

    if not publications:
        print(f"No more publications found at {page_url}")
        return False  # No publications found, stop the loop

    for pub in publications:
        title = pub.text.strip()  # Keeping the original title as filename
        pub_url = pub["href"] if pub.has_attr("href") else None

        if pub_url:
            print(f"Title: {title}")
            print(f"Publication URL: {pub_url}")

            # Visit the publication page to extract both image and PDF
            pdf_url, image_url = get_image_and_pdf_from_publication(pub_url, title)

            # Download PDF if found
            if pdf_url:
                pdf_filename = f"{title}.pdf"  # Use original title for PDF
                print(f"PDF URL: {pdf_url}\n")
                download_file(pdf_url, "cfa_pdfs", pdf_filename)

            # Download image if found
            if image_url:
                image_filename = f"{title}.jpg"  # Use original title for the image
                print(f"Image URL: {image_url}\n")
                download_file(image_url, "cfa_images", image_filename)

    return True  # Publications found, continue to the next page

def scrape_all_pages():
    """Scrape all pages dynamically until no more results are found."""
    base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
    page_number = 0

    while True:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")

        # Scrape the current page
        if not scrape_page(page_url):
            break  # Stop if no more publications are found

        page_number += 1

# Run the scraper
scrape_all_pages()

# Quit the WebDriver
driver.quit()