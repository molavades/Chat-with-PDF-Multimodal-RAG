import os
import re
import requests
import boto3
import snowflake.connector
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pymupdf as fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Snowflake and AWS credentials from environment variables
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_TABLE = os.getenv("SNOWFLAKE_TABLE")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
bucket_name = os.getenv("BUCKET_NAME")
pdf_folder = os.getenv("PDF_FOLDER")
image_folder = os.getenv("IMAGE_FOLDER")

# Configure Selenium (headless mode for faster scraping)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize Boto3 S3 client with credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def setup_snowflake():
    """Setup schema and table in Snowflake, clear old data if it exists."""
    conn = None
    cursor = None
    try:
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        
        cursor = conn.cursor()
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
        
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE} (
            title STRING,
            brief_summary STRING,
            pdf_link STRING,
            image_link STRING
        );
        """
        cursor.execute(create_table_query)
        print(f"Table '{SNOWFLAKE_TABLE}' checked/created successfully in Snowflake.")
        
        delete_query = f"DELETE FROM {SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE}"
        cursor.execute(delete_query)
        print(f"Old data in '{SNOWFLAKE_TABLE}' has been cleared.")
        
    except Exception as e:
        print(f"Failed to set up schema/table in Snowflake. Error: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

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

def insert_into_snowflake(title, pdf_link, image_link, brief_summary):
    """Insert publication details into the Snowflake table."""
    conn = None
    cursor = None
    try:
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        
        cursor = conn.cursor()
        insert_query = f"""
            INSERT INTO {SNOWFLAKE_SCHEMA}.{SNOWFLAKE_TABLE} (title, pdf_link, image_link, brief_summary)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (title, pdf_link, image_link, brief_summary))
        conn.commit()
        print(f"Inserted record for {title} into Snowflake.")
        
    except Exception as e:
        print(f"Failed to insert record for {title} into Snowflake. Error: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_pdf_from_publication(publication_url):
    """Extract the PDF link from the publication page."""
    try:
        driver.get(publication_url)
        time.sleep(5)
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
    """Scrape all publications on a specific page to download PDFs and capture brief summary."""
    driver.get(page_url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "lxml")
    publications = soup.select(".coveo-result-cell")

    if not publications:
        print(f"No more publications found at {page_url}")
        return False

    new_publication_found = False

    for pub in publications:
        title_elem = pub.select_one(".CoveoResultLink")
        body_elem = pub.select_one(".result-body")

        if title_elem:
            title = title_elem.text.strip()
            pub_url = title_elem["href"]

            # Print title and publication URL once
            print(f"Title: {title}")
            print(f"Publication URL: {pub_url}")

            # Extract and clean brief summary text
            brief_summary = body_elem.text.strip() if body_elem else "No description available"

            cleaned_title = clean_title(title)
            pdf_url = get_pdf_from_publication(pub_url)

            if pdf_url and pdf_url not in downloaded_pdfs:
                pdf_filename = f"{cleaned_title}.pdf"
                pdf_s3_link = download_file_and_upload_to_s3(pdf_url, pdf_folder, pdf_filename)
                image_s3_link = convert_pdf_first_page_to_image_and_upload(pdf_url, cleaned_title)

                if pdf_s3_link and image_s3_link:
                    # Insert both title and brief_summary into Snowflake
                    insert_into_snowflake(title, pdf_s3_link, image_s3_link, brief_summary)
                
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

    # Ensure Snowflake is set up with the necessary schema and table
    setup_snowflake()

    while True:
        page_url = f"{base_url}#first={page_number * 10}&sort=%40officialz32xdate%20descending"
        print(f"Scraping: {page_url}")
        if not scrape_page_for_pdfs(page_url, downloaded_pdfs):
            break
        page_number += 1

# Wrap the main function in an entry point
if __name__ == "__main__":
    # for local run
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    # for docker container run
    driver = webdriver.Remote(command_executor='http://selenium-chrome:4444/wd/hub', options=chrome_options)
    try:
        scrape_all_pages_for_pdfs()
    finally:
        driver.quit()