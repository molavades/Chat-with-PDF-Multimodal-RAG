import os
import re
import requests
import boto3
import csv
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

# Initialize Boto3 S3 client
s3 = boto3.client('s3')

# Define your S3 bucket name and folders
bucket_name = "cfa-bigdata"
pdf_folder = "cfa_pdfs/"
image_folder = "cfa_images/"

# Define CSV file
csv_file = "publications_data.csv"

# Create the CSV file and write headers
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "PDF_Link", "Image_Link"])

def clean_title(title):
    """Remove non-word characters except spaces, allowing Chinese and alphabetic characters."""
    return re.sub(r'[^\w\s]', '', title, flags=re.UNICODE).strip()

def upload_to_s3(content, folder, filename):
    """Upload the file content to S3 and return the S3 URL."""
    try:
        s3.put_object(Bucket=bucket_name, Key=f"{folder}{filename}", Body=content)
        s3_link = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{folder}{filename}"
        print(f"Uploaded {filename} to {s3_link}")
        return s3_link
    except Exception as e:
        print(f"Failed to upload {filename} to S3. Error: {e}")
        return None

def download_file_and_upload_to_s3(url, folder, filename):
    """Download the file from the given URL and upload it to S3."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return upload_to_s3(response.content, folder, filename)
    except Exception as e:
        print(f"Failed to download and upload {filename}. Error: {e}")
        return None

def convert_pdf_first_page_to_image_and_upload(pdf_url, filename):
    """Convert the first page of a PDF to an image, upload to S3, and return S3 link."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_content = response.content

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        first_page = pdf_document.load_page(0)
        pix = first_page.get_pixmap()
        image_bytes = pix.tobytes("png")
        pdf_document.close()

        return upload_to_s3(image_bytes, image_folder, f"{filename}.png")
    except Exception as e:
        print(f"Failed to convert PDF {filename} to image and upload to S3. Error: {e}")
        return None

def get_pdf_from_publication(publication_url):
    """Extract the PDF link from the publication page."""
    try:
        driver.get(publication_url)
        time.sleep(3)
        publication_soup = BeautifulSoup(driver.page_source, "lxml")
        base_url = "https://rpc.cfainstitute.org"
        pdf_url = None
        links = publication_soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if href.lower().endswith(".pdf"):
                if not href.startswith("http"):
                    href = f"{base_url}{href}"
                pdf_url = href
                break
        return pdf_url
    except Exception as e:
        print(f"Error accessing {publication_url}: {e}")
        return None

def scrape_page_for_pdfs(page_url, downloaded_pdfs):
    """Scrape all publications on a specific page to download PDFs."""
    driver.get(page_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    publications = soup.select(".CoveoResultLink")

    if not publications:
        print(f"No more publications found at {page_url}")
        return False

    new_publication_found = False

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for pub in publications:
            title = pub.text.strip()
            pub_url = pub["href"] if pub.has_attr("href") else None

            if pub_url:
                print(f"Title: {title}")
                print(f"Publication URL: {pub_url}")
                cleaned_title = clean_title(title)
                pdf_url = get_pdf_from_publication(pub_url)

                if pdf_url and pdf_url not in downloaded_pdfs:
                    pdf_filename = f"{cleaned_title}.pdf"
                    pdf_s3_link = download_file_and_upload_to_s3(pdf_url, pdf_folder, pdf_filename)
                    image_s3_link = convert_pdf_first_page_to_image_and_upload(pdf_url, cleaned_title)

                    # Write to CSV
                    if pdf_s3_link and image_s3_link:
                        writer.writerow([title, pdf_s3_link, image_s3_link])
                    
                    downloaded_pdfs.add(pdf_url)
                    new_publication_found = True

    if not new_publication_found:
        print("No new publications found on this page.")
        return False

    return True

def scrape_all_pages_for_pdfs():
    """Scrape all pages dynamically until no more results are found."""
    base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
    page_number = 0
    downloaded_pdfs = set()

    while True:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")
        if not scrape_page_for_pdfs(page_url, downloaded_pdfs):
            break
        page_number += 1

scrape_all_pages_for_pdfs()
driver.quit()
