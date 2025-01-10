import os
import re
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import fitz  # PyMuPDF for PDF processing

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
    return file_path  # Return the file path for further processing

def download_file(url, folder, filename):
    """Download the file from the given URL and save it locally."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return save_file(response.content, folder, filename)
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")
        return None

def convert_pdf_first_page_to_image(pdf_path, image_folder, image_filename):
    """Convert the first page of a PDF to an image and save it."""
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        # Get the first page
        first_page = pdf_document.load_page(0)
        # Render the page as an image (pixmap)
        pix = first_page.get_pixmap()
        # Save the image as PNG
        image_path = os.path.join(image_folder, f"{image_filename}.png")
        pix.save(image_path)
        print(f"Saved first page of {pdf_path} as an image in {image_path}")
        pdf_document.close()
    except Exception as e:
        print(f"Failed to convert {pdf_path} to image. Error: {e}")

def get_pdf_from_publication(publication_url):
    """Extract the PDF link from the publication page."""
    try:
        driver.get(publication_url)
        time.sleep(3)  # Give the page some time to load

        publication_soup = BeautifulSoup(driver.page_source, "lxml")
        
        # Define base URL to handle relative paths
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

        return pdf_url

    except Exception as e:
        print(f"Error accessing {publication_url}: {e}")
        return None

def scrape_page_for_pdfs(page_url, downloaded_pdfs):
    """Scrape all publications on a specific page to download PDFs."""
    driver.get(page_url)
    time.sleep(3)  # Wait for the page to load

    soup = BeautifulSoup(driver.page_source, "lxml")
    publications = soup.select(".CoveoResultLink")

    if not publications:
        print(f"No more publications found at {page_url}")
        return False  # No publications found, stop the loop

    new_publication_found = False  # Track if any new publication is found on the page

    for pub in publications:
        title = pub.text.strip()  # Keeping the original title as filename
        pub_url = pub["href"] if pub.has_attr("href") else None

        if pub_url:
            print(f"Title: {title}")
            print(f"Publication URL: {pub_url}")

            # Clean the title to keep only alphabetic characters for the filename
            cleaned_title = clean_title(title)

            # Visit the publication page to extract the PDF link
            pdf_url = get_pdf_from_publication(pub_url)

            # Download PDF if found and not already downloaded
            if pdf_url and pdf_url not in downloaded_pdfs:
                pdf_filename = f"{cleaned_title}.pdf"  # Use cleaned title for PDF
                print(f"PDF URL: {pdf_url}\n")
                
                # Download the PDF file
                pdf_path = download_file(pdf_url, "cfa_pdfs", pdf_filename)
                
                # Convert the first page of the PDF to an image if download succeeded
                if pdf_path:
                    image_filename = cleaned_title  # Use cleaned title for image filename
                    convert_pdf_first_page_to_image(pdf_path, "cfa_images", image_filename)
                
                downloaded_pdfs.add(pdf_url)  # Add the URL to the set to avoid duplicates
                new_publication_found = True

    if not new_publication_found:
        print("No new publications found on this page.")
        return False  # Stop if no new publications were found

    return True  # New publications found, continue to the next page

def scrape_all_pages_for_pdfs():
    """Scrape all pages dynamically until no more results are found."""
    base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
    page_number = 0
    downloaded_pdfs = set()  # Set to keep track of downloaded PDF URLs

    while True:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")

        # Scrape the current page for PDFs
        if not scrape_page_for_pdfs(page_url, downloaded_pdfs):
            break  # Stop if no more publications are found or no new publications found

        page_number += 1

# Run the scraper to download PDFs and convert the first page to images
scrape_all_pages_for_pdfs()

# Quit the WebDriver
driver.quit()