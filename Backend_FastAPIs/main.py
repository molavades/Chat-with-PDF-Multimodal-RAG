from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from psycopg2 import connect, sql, OperationalError, errors
from uuid import uuid4, UUID
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Union, Any, AsyncGenerator, Set, Tuple, Dict
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2.service_account import Credentials
from contextlib import asynccontextmanager
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import snowflake.connector
import requests
import io
import fitz
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import traceback
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

from urllib.parse import urlparse
from fastapi import HTTPException
from fastapi import BackgroundTasks
import asyncio
import base64
import tempfile
import logging
import os
import glob
import shutil
import tempfile
from pathlib import Path
import logging
from typing import List, Set
import asyncio
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, List, Any
from llama_index.core import Settings, SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
import nest_asyncio
import uuid
import uuid as uuid_pkg
import shutil
import tempfile
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Document, VectorStoreIndex
from PIL import Image
import io
import base64
from pathlib import Path
import logging
import tempfile
import uuid
import os
import shutil
from typing import Dict, List, Optional
from fastapi import HTTPException, BackgroundTasks

# Initialize FastAPI app

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start cleanup task
    start_cleanup_task()
    
    # Initialize database
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(CREATE_USERS_TABLE_QUERY)
        conn.commit()
    finally:
        cur.close()
        conn.close()
    
    # Initialize RAG settings
    initialize_rag_settings()
    
    # Set up ThreadPoolExecutor for async operations
    app.state.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    
    yield
    
    # Cleanup on shutdown
    await cleanup_temp_files()
    
    # Shutdown executor
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_temp_directory(prefix="report_gen_"):
    """Create a temporary directory with proper cleanup handling."""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return temp_dir

async def process_image_async(image_path: str) -> Optional[str]:
    """Process an image file and return its base64 encoding."""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            # Optimize image size if needed
            img = Image.open(BytesIO(img_data))
            if img.size[0] > 1200:  # If width > 1200px
                ratio = 1200 / img.size[0]
                new_size = (1200, int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85, optimize=True)
                img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

from typing import List, Optional, Union
from pydantic import BaseModel

# Define block classes before using them
# class TextBlock(BaseModel):
#     """Model for text content in reports."""
#     content: str
#     order: Optional[int] = None

# class ImageBlock(BaseModel):
#     """Model for image content in reports."""
#     path: str
#     caption: Optional[str] = None
#     order: Optional[int] = None

# def validate_report_blocks(blocks: List[Union[TextBlock, ImageBlock]]) -> bool:
#     """Validate that report blocks meet minimum requirements."""
#     if not blocks:
#         return False
#     has_text = any(isinstance(block, TextBlock) for block in blocks)
#     has_image = any(isinstance(block, ImageBlock) for block in blocks)
#     return has_text and has_image

class ReportGenerator:
    """Helper class to manage report generation process."""
    
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
        self.temp_directories = set()
    
    async def cleanup(self):
        """Clean up any temporary directories created during report generation."""
        for temp_dir in self.temp_directories:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {temp_dir}: {str(e)}")
        self.temp_directories.clear()
    
    async def process_documents(self, documents: List[str]) -> Tuple[List[Document], Dict]:
        """Process multiple documents and return nodes and image data."""
        temp_dir = setup_temp_directory()
        self.temp_directories.add(temp_dir)
        
        try:
            return await process_documents_with_llamaparse(documents, temp_dir)
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def create_system_prompt(self, available_images: Dict) -> str:
        """Create a system prompt based on available images."""
        image_info = "\n".join([
            f"- Image at {path}: {meta.get('caption', 'No caption')}"
            for path, meta in available_images.items()
        ])
        
        return f"""You are a report generation assistant tasked with creating detailed, 
        well-formatted reports that combine text and images effectively.

        Available images:
        {image_info}

        Guidelines:
        1. Include relevant images that support your text
        2. Add clear captions to explain each image
        3. Maintain a logical flow between text and images
        4. Use markdown formatting for text when appropriate
        5. Cite specific data points and findings from the documents
        """

class TempFileManager:
    """Manages temporary files and directories for the application."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "multimodal_rag"
        self.temp_dir.mkdir(exist_ok=True)
        self.active_files: Set[Path] = set()
        
    def create_temp_file(self, suffix: str = None) -> Path:
        """Create a new temporary file and track it."""
        temp_file = Path(tempfile.mktemp(dir=str(self.temp_dir), suffix=suffix))
        self.active_files.add(temp_file)
        return temp_file
    
    def create_temp_dir(self) -> Path:
        """Create a new temporary directory and track it."""
        temp_dir = Path(tempfile.mkdtemp(dir=str(self.temp_dir)))
        self.active_files.add(temp_dir)
        return temp_dir
    
    def release_file(self, file_path: Path):
        """Mark a file as no longer needed."""
        try:
            if file_path in self.active_files:
                self.active_files.remove(file_path)
        except Exception as e:
            logger.error(f"Error releasing file {file_path}: {str(e)}")

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files and directories."""
        try:
            # Clean up tracked files first
            for file_path in list(self.active_files):
                try:
                    if file_path.is_file():
                        os.unlink(str(file_path))
                    elif file_path.is_dir():
                        shutil.rmtree(str(file_path))
                    self.active_files.remove(file_path)
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {str(e)}")

            # Clean up old files in temp directory
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up files
            for file_path in self.temp_dir.glob("**/*"):
                try:
                    if file_path.is_file():
                        # Check file age
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_time:
                            os.unlink(str(file_path))
                except Exception as e:
                    logger.error(f"Error cleaning up old file {file_path}: {str(e)}")

            # Clean up empty directories
            for dirpath, dirnames, filenames in os.walk(str(self.temp_dir), topdown=False):
                if not dirnames and not filenames and dirpath != str(self.temp_dir):
                    try:
                        os.rmdir(dirpath)
                    except Exception as e:
                        logger.error(f"Error removing empty directory {dirpath}: {str(e)}")

        except Exception as e:
            logger.error(f"Error during temp file cleanup: {str(e)}")

# Create a global instance of the temp file manager
temp_file_manager = TempFileManager()

async def cleanup_temp_files():
    """Wrapper function for the temp file manager's cleanup method."""
    await temp_file_manager.cleanup_temp_files()

def get_temp_file_path(suffix: str = None) -> Path:
    """Get a new temporary file path."""
    return temp_file_manager.create_temp_file(suffix)

def get_temp_dir_path() -> Path:
    """Get a new temporary directory path."""
    return temp_file_manager.create_temp_dir()

def release_temp_file(file_path: Path):
    """Release a temporary file when it's no longer needed."""
    temp_file_manager.release_file(file_path)

# Periodic cleanup task
async def periodic_cleanup():
    """Run cleanup periodically."""
    while True:
        try:
            await cleanup_temp_files()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")
            await asyncio.sleep(60)  # Wait a minute before retrying

# Function to start the periodic cleanup task
def start_cleanup_task():
    """Start the periodic cleanup task."""
    asyncio.create_task(periodic_cleanup())

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configuration variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        if len(v) < 3 or len(v) > 30:
            raise ValueError('Username must be between 3 and 30 characters')
        return v

    @field_validator('password')
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]

class ProcessingStatus(BaseModel):
    status: str
    document_count: int

class DocumentInfo(BaseModel):
    title: str
    brief_summary: str
    pdf_link: str
    image_link: str

class AnalyzeDocumentsRequest(BaseModel):
    pdf_links: List[str]
    analysis_type: str  # 'summary' or 'qa'
    question: Optional[str] = None

# Database table creation
CREATE_USERS_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

# Initialize security components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# # Initialize RAG settings
# def initialize_rag_settings():
#     Settings.llm = NVIDIA(
#         model="meta/llama3-70b-instruct",
#         api_key=NVIDIA_API_KEY
#     )
#     Settings.embed_model = NVIDIAEmbedding(
#         model="nvidia/nv-embedqa-e5-v5",
#         api_key=NVIDIA_API_KEY
#     )
async def get_pdf_from_s3_async(pdf_url: str) -> BytesIO:
    """Asynchronously download PDF from S3 and return as BytesIO object"""
    try:
        # Parse the S3 URL to get bucket and key
        parsed_url = urlparse(pdf_url)
        bucket = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')
        
        # Get the object from S3
        loop = asyncio.get_event_loop()
        s3_client = get_s3_client()
        
        # Run S3 get_object in thread pool since boto3 is not async
        response = await loop.run_in_executor(
            None,
            lambda: s3_client.get_object(Bucket=bucket, Key=key)
        )
        
        # Read the PDF content
        pdf_content = await loop.run_in_executor(
            None,
            lambda: response['Body'].read()
        )
        
        return BytesIO(pdf_content)
    except ClientError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Error accessing PDF from S3: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
async def get_json_result_async(self, file_path: str) -> List[Any]:
    """Asynchronously get JSON result from file"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        self.get_json_result,
        file_path
    )

async def get_images_async(self, json_objs: List[Any], download_path: str) -> dict:
    """Asynchronously get images from JSON objects"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        self.get_images,
        json_objs,
        download_path
    )

def initialize_rag_settings():
    Settings.llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )
    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )


# Database functions
def get_db_connection():
    try:
        return connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except OperationalError:
        raise HTTPException(status_code=500, detail="Database connection error")

def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
   
# def get_pdf_from_s3(pdf_url: str) -> BytesIO:
#     """Download PDF from S3 and return as BytesIO object"""
#     try:
#         # Parse the S3 URL correctly for https format
#         parsed_url = urlparse(pdf_url)
#         # Extract bucket name from the hostname (first part before first dot)
#         bucket = parsed_url.netloc.split('.')[0]
#         # Remove leading slash and decode URL-encoded characters from the path
#         key = parsed_url.path.lstrip('/')
        
#         # For debugging
#         print(f"Bucket: {bucket}")
#         print(f"Key: {key}")
        
#         # Get the object from S3
#         s3_client = get_s3_client()
#         try:
#             response = s3_client.get_object(Bucket=bucket, Key=key)
#             return BytesIO(response['Body'].read())
#         except ClientError as e:
#             error_code = e.response.get('Error', {}).get('Code', 'Unknown')
#             error_message = e.response.get('Error', {}).get('Message', str(e))
#             raise HTTPException(
#                 status_code=404 if error_code == 'NoSuchKey' else 500,
#                 detail=f"S3 Error ({error_code}): {error_message}"
#             )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing PDF: {str(e)}"
#         )[1][4]

def get_pdf_from_s3(pdf_url: str) -> BytesIO:
    """Download PDF from S3 and return as BytesIO object"""
    try:
        # Parse the S3 URL to get bucket and key
        parsed_url = urlparse(pdf_url)
        bucket = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')
        
        # Get the object from S3
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read the PDF content
        pdf_content = response['Body'].read()
        return BytesIO(pdf_content)
    except ClientError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Error accessing PDF from S3: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

# Authentication functions
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, hashed_password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if user and verify_password(password, user[1]):
            return {"id": user[0], "username": username}
        return None
    except Exception:
        raise HTTPException(status_code=500, detail="Error during authentication")
    finally:
        cur.close()
        conn.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Authentication endpoints
@app.post("/register", response_model=UserOut)
def register_user(user: UserCreate):
    try:
        hashed_password = hash_password(user.password)
        user_id = uuid4()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (id, email, username, hashed_password) VALUES (%s, %s, %s, %s) RETURNING created_at",
            (str(user_id), user.email, user.username, hashed_password))
        created_at = cur.fetchone()[0]
        conn.commit()
        return {"id": user_id, "email": user.email, "username": user.username, "created_at": created_at}
    except errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Email or username already exists")
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error during registration: {str(e)}")
    finally:
        cur.close()
        conn.close()

def process_document_with_chunking(pdf_stream):
    """Process a PDF document and return chunked text"""
    # Extract text using PyMuPDF
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    full_text = ""
    for page in pdf_document:
        full_text += page.get_text()
    pdf_document.close()
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust this value based on your needs
        chunk_overlap=50,  # Some overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(full_text)
    return chunks

@app.post("/analyze-documents", response_model=QueryResponse)
async def analyze_documents(
    request: AnalyzeDocumentsRequest,
    token: str = Depends(oauth2_scheme)
):
    """Analyze multiple documents with either summary or Q&A"""
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")

        # Validate analysis type
        if request.analysis_type not in ["summary", "qa"]:
            raise HTTPException(status_code=400, detail="Invalid analysis_type. Must be 'summary' or 'qa'")
        
        if request.analysis_type == "qa" and not request.question:
            raise HTTPException(status_code=400, detail="Question required for Q&A analysis")

        # Process documents
        all_chunks = []
        for pdf_link in request.pdf_links:
            try:
                # Get PDF from S3
                pdf_stream = get_pdf_from_s3(pdf_link)
                
                # Process document and get chunks
                chunks = process_document_with_chunking(pdf_stream)
                
                # Create Document objects for each chunk
                doc_chunks = [
                    Document(
                        text=chunk,
                        metadata={
                            "source": pdf_link,
                            "chunk_index": i
                        }
                    ) for i, chunk in enumerate(chunks)
                ]
                all_chunks.extend(doc_chunks)
                
            except Exception as e:
                print(f"Error processing PDF {pdf_link}: {str(e)}")
                continue

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No documents could be processed")

        # Create index with chunked documents
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dim=1024,
            collection_name="document_store"
        )
        
        index = VectorStoreIndex.from_documents(
            all_chunks,
            vector_store=vector_store
        )
        
        # Configure query engine with appropriate parameters
        query_engine = index.as_query_engine(
            similarity_top_k=5,  # Increased to get more context
            streaming=True
        )

        # Generate prompt based on analysis type
        if request.analysis_type == "summary":
            prompt = """Please provide a comprehensive summary of these documents. 
            Focus on the main points and key findings. Structure the summary in a clear and organized way."""
        else:  # qa
            prompt = request.question

        # Get response
        response = query_engine.query(prompt)
        
        # Process source documents
        source_docs = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_docs.append({
                    "source": node.metadata.get("source", "Unknown"),
                    "title": node.metadata.get("title", "Unknown"),
                    "relevance_score": float(node.score) if hasattr(node, 'score') else 0.0
                })
        
        return {
            "answer": str(response),
            "source_documents": source_docs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not form_data.username or not form_data.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user['username']}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserOut)
async def read_user_me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, username, created_at FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": UUID(user[0]), "email": user[1], "username": user[2], "created_at": user[3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user information: {str(e)}")
    finally:
        cur.close()
        conn.close()

# RAG endpoints
@app.post("/process-documents", response_model=ProcessingStatus)
async def process_documents(token: str = Depends(oauth2_scheme)):
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")

        # Initialize vector store
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dim=1024,
            collection_name="document_store"
        )

        # Get documents from Snowflake
        documents = []
        snow_conn = get_snowflake_connection()
        try:
            cur = snow_conn.cursor()
            cur.execute(f"SELECT * FROM {os.getenv('SNOWFLAKE_TABLE')}")
            for row in cur:
                pdf_url = row[3]
                try:
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        # Extract text using PyMuPDF
                        memory_stream = io.BytesIO(response.content)
                        pdf_document = fitz.open(stream=memory_stream, filetype="pdf")
                        text = ""
                        for page in pdf_document:
                            text += page.get_text()
                        
                        doc = Document(
                            text=text,
                            metadata={
                                "title": row[0],
                                "source": pdf_url,
                            }
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing PDF {pdf_url}: {str(e)}")
                    continue
        finally:
            snow_conn.close()

        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")

        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store
        )
        
        return {
            "status": "success",
            "document_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query: QueryRequest,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")

        # Initialize vector store
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dim=1024,
            collection_name="document_store"
        )
        
        # Create index and query engine
        index = VectorStoreIndex(vector_store=vector_store)
        query_engine = index.as_query_engine(
            similarity_top_k=3
        )
        
        # Get response
        response = query_engine.query(query.question)
        
        # Process source documents
        source_docs = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_docs.append({
                    "source": node.metadata.get("source", "Unknown"),
                    "title": node.metadata.get("title", "Unknown"),
                    "relevance_score": float(node.score) if hasattr(node, 'score') else 0.0
                })
        
        return {
            "answer": str(response),
            "source_documents": source_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents(token: str = Depends(oauth2_scheme)):
    """Get list of available documents from Snowflake"""
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get documents from Snowflake
        snow_conn = get_snowflake_connection()
        try:
            cur = snow_conn.cursor()
            cur.execute(f"SELECT title, brief_summary, pdf_link, image_link FROM {os.getenv('SNOWFLAKE_TABLE')}")
            documents = []
            for row in cur:
                documents.append({
                    "title": row[0],
                    "brief_summary": row[1],
                    "pdf_link": row[2],
                    "image_link": row[3]
                })
            return documents
        finally:
            snow_conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_documents(
    pdf_links: List[str],
    token: str = Depends(oauth2_scheme)
):
    """Summarize selected documents"""
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")

        # Initialize RAG components
        documents = []
        for pdf_link in pdf_links:
            try:
                # Get PDF from S3
                pdf_stream = get_pdf_from_s3(pdf_link)
                
                # Extract text using PyMuPDF
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()
                
                # Close the PDF document
                pdf_document.close()
                
                doc = Document(
                    text=text,
                    metadata={"source": pdf_link}
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error processing PDF {pdf_link}: {str(e)}")
                continue

        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")

        # Create index
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dim=1024,
            collection_name="summary_store"
        )
        
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store
        )
        
        # Generate summary
        query_engine = index.as_query_engine()
        response = query_engine.query(
            "Please provide a comprehensive summary of these documents, highlighting the main points and key findings."
        )
        
        return {"summary": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-content")
async def get_document_content(
    pdf_link: str,
    token: str = Depends(oauth2_scheme)
):
    """Get content of a specific document"""
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get PDF content from S3
        try:
            pdf_stream = get_pdf_from_s3(pdf_link)
            
            # Extract text using PyMuPDF
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            content = ""
            for page in pdf_document:
                content += page.get_text()
            
            # Get total pages before closing the document
            total_pages = len(pdf_document)
            
            # Close the PDF document
            pdf_document.close()
            
            return {
                "content": content,
                "total_pages": total_pages
            }
        except Exception as e:
            traceback.print_exc()  # Add this line for detailed exception logging
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
            
    except Exception as e:
        traceback.print_exc()  # Add this line for detailed exception logging
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )
    
# Health check endpoint
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "services": {
            "database": "unknown",
            "snowflake": "unknown",
            "milvus": "unknown"
        }
    }
    
    # Check database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Snowflake
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health_status["services"]["snowflake"] = "healthy"
    except Exception as e:
        health_status["services"]["snowflake"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Milvus
    try:
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dim=1024,
            collection_name="document_store"
        )
        health_status["services"]["milvus"] = "healthy"
    except Exception as e:
        health_status["services"]["milvus"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


# Add after existing Pydantic models
from pydantic import BaseModel, Field
from typing import List, Union

class TextBlock(BaseModel):
    """Text block."""
    text: str = Field(..., description="The text for this block.")

class ImageBlock(BaseModel):
    """Image block with base64 encoding."""
    image_base64: str = Field(..., description="Base64-encoded image data")
    caption: Optional[str] = Field(None, description="Optional caption for the image")

class ReportOutput(BaseModel):
    """Data model for a report that can contain both text and images."""
    blocks: List[TextBlock | ImageBlock] = Field(
        ..., 
        description="A list of text and image blocks"
    )

# Replace your existing AnalyzeDocumentsRequest model with this one
class AnalyzeDocumentsRequest(BaseModel):
    """Model for document analysis request."""
    pdf_links: List[str]
    question: str


# Add after existing helper functions
import re
from pathlib import Path

def get_page_number(file_name):
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

def _get_sorted_image_files(image_dir):
    """Get image files sorted by page."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files

def get_text_nodes(json_dicts, image_dir=None):
    """Create text nodes with image metadata."""
    nodes = []

    image_files = _get_sorted_image_files(image_dir) if image_dir is not None else None
    md_texts = [d["md"] for d in json_dicts]

    for idx, md_text in enumerate(md_texts):
        chunk_metadata = {"page_num": idx + 1}
        if image_files is not None and idx < len(image_files):
            image_file = image_files[idx]
            chunk_metadata["image_path"] = str(image_file)
        chunk_metadata["parsed_text_markdown"] = md_text
        node = Document(
            text="",  # Text is empty because we're using metadata
            metadata=chunk_metadata,
        )
        nodes.append(node)

    return nodes


import base64
from pathlib import Path
import re
from llama_index.core import Settings, SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio

nest_asyncio.apply()
# Add after existing endpoint definitions
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI


LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

async def process_documents_with_llamaparse(pdf_paths: List[str], download_path: str):
    """Process documents with LlamaParse and handle image processing."""
    try:
        parser = LlamaParse(
            api_key=LLAMA_PARSE_API_KEY,
            result_type="markdown",
            use_vendor_multimodal_model=True,  # Enable multimodal processing
            vendor_multimodal_model_name="anthropic-sonnet-3.5",  # Specify model
            timeout=300
        )
        
        # Create directories
        base_dir = Path("data_images")
        download_dir = base_dir / download_path
        base_dir.mkdir(exist_ok=True)
        download_dir.mkdir(exist_ok=True)
        
        all_text_nodes = []
        all_images = {}
        
        for pdf_idx, pdf_path in enumerate(pdf_paths):
            try:
                # Get JSON result
                json_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    parser.get_json_result,
                    pdf_path
                )
                
                if not json_result or not json_result[0].get("pages"):
                    logging.error(f"No pages found in document {pdf_path}")
                    continue
                
                job_id = json_result[0].get("job_id")
                if not job_id:
                    logging.error(f"No job ID found for document {pdf_path}")
                    continue
                
                # Process each page
                for page_idx, page in enumerate(json_result[0]["pages"], 1):
                    # Generate unique image filename
                    img_filename = f"doc_{pdf_idx}_page_{page_idx}.jpg"
                    img_path = str(download_dir / img_filename)
                    
                    try:
                        # Get image from LlamaParse API
                        image_url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/image/page_{page_idx}.jpg"
                        headers = {"Authorization": f"Bearer {LLAMA_PARSE_API_KEY}"}
                        
                        img_response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: requests.get(image_url, headers=headers)
                        )
                        
                        if img_response.status_code == 200:
                            # Save and optimize image
                            img_data = img_response.content
                            img = Image.open(io.BytesIO(img_data))
                            
                            # Resize if too large
                            if img.size[0] > 1200:
                                ratio = 1200 / img.size[0]
                                new_size = (1200, int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # Save optimized image
                            img.save(img_path, "JPEG", quality=85, optimize=True)
                            
                            # Convert to base64
                            with open(img_path, "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                            
                            # Store image metadata
                            all_images[img_path] = {
                                "page": page_idx,
                                "source_pdf": pdf_path,
                                "type": "full_page",
                                "base64_data": base64_data,
                                "text_context": page.get("md", "")[:200]  # Store some context
                            }
                            
                            logging.info(f"Successfully processed image: {img_path}")
                        else:
                            logging.error(f"Failed to download image for page {page_idx}: {img_response.status_code}")
                            continue
                            
                    except Exception as e:
                        logging.error(f"Error processing image for page {page_idx}: {str(e)}")
                        continue
                    
                    # Create text node with image metadata
                    node_metadata = {
                        "page_num": page_idx,
                        "source_pdf": pdf_path,
                        "parsed_text_markdown": page.get("md", ""),
                        "image_path": img_path
                    }
                    
                    node = Document(
                        text=page.get("md", ""),
                        metadata=node_metadata,
                    )
                    all_text_nodes.append(node)
                
            except Exception as e:
                logging.error(f"Error processing document {pdf_path}: {str(e)}")
                continue
        
        return all_text_nodes, all_images
    
    except Exception as e:
        logging.error(f"Error in LlamaParse processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents with LlamaParse: {str(e)}"
        )

@app.post("/generate-report", response_model=ReportOutput)
async def generate_report(
    request: AnalyzeDocumentsRequest,
    token: str = Depends(oauth2_scheme),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate a multimodal report with properly embedded images."""
    pdf_paths = []
    temp_dir = None
    
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")

        # Create temporary directory with unique ID
        unique_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp(prefix=f"report_gen_{unique_id}_")
        
        # Process documents and get PDF paths
        for pdf_link in request.pdf_links:
            try:
                pdf_stream = await get_pdf_from_s3_async(pdf_link)
                temp_file = Path(temp_dir) / f"{uuid.uuid4()}.pdf"
                temp_file.write_bytes(pdf_stream.getvalue())
                pdf_paths.append(str(temp_file))
            except Exception as e:
                logging.error(f"Error processing document {pdf_link}: {str(e)}")

        # Process documents with LlamaParse
        all_text_nodes, all_images = await process_documents_with_llamaparse(
            pdf_paths,
            download_path=unique_id
        )

        if not all_text_nodes:
            raise HTTPException(status_code=400, detail="No documents could be processed")

        # Create report blocks list to store content
        report_blocks = []

        # First, add introductory text block
        intro_text = """# Document Analysis Report
        
This report provides an analysis of the uploaded documents with relevant visualizations and insights.
"""
        report_blocks.append(TextBlock(text=intro_text))

        # Create index and query engine with custom prompt
        index = VectorStoreIndex(nodes=all_text_nodes)
        
        system_prompt = """You are a report generation assistant creating well-formatted reports
        combining text and images. Follow these guidelines:
        1. Analyze the content thoroughly
        2. When mentioning visual content, explicitly reference 'the image below' or 'the figure above'
        3. Write clear, analytical insights
        4. Break down complex concepts
        5. Use markdown formatting for better readability
        6. Keep your analysis focused and relevant"""
        
        llm = OpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4",
            temperature=0.3,
            system_prompt=system_prompt
        )
        
        # Configure query engine
        query_engine = index.as_query_engine(
            response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True
            ),
            streaming=True
        )

        # Generate initial content
        response = await query_engine.aquery(request.question)
        
        # Add main content text block
        if response.response:
            report_blocks.append(TextBlock(text=str(response.response)))

        # Process and add images with proper formatting
        if hasattr(response, 'source_nodes'):
            for idx, node in enumerate(response.source_nodes):
                if 'image_path' in node.metadata and node.metadata['image_path'] in all_images:
                    img_path = node.metadata['image_path']
                    img_data = all_images[img_path]
                    
                    # Create descriptive caption
                    caption = f"Figure {idx + 1}: Page {img_data['page']} from document"
                    if 'parsed_text_markdown' in node.metadata:
                        # Add a brief excerpt from the text content
                        excerpt = node.metadata['parsed_text_markdown'][:100] + "..."
                        caption += f"\n\nContext: {excerpt}"

                    # Add image block with caption
                    report_blocks.append(ImageBlock(
                        image_base64=img_data['base64_data'],
                        caption=caption
                    ))

                    # Add explanatory text after each image
                    analysis_text = f"""
### Analysis of Figure {idx + 1}

This visualization shows content from page {img_data['page']} of the document. 
Key points from this section include:

- {node.metadata.get('parsed_text_markdown', 'Content analysis unavailable.').split('.')[0]}.
"""
                    report_blocks.append(TextBlock(text=analysis_text))

        # Add conclusion text block
        conclusion_text = """
## Conclusion

This report has presented key visualizations and analysis from the provided documents. 
Each figure has been analyzed in context to provide meaningful insights.
"""
        report_blocks.append(TextBlock(text=conclusion_text))

        return ReportOutput(blocks=report_blocks)

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for pdf_path in pdf_paths:
            try:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            except Exception as e:
                logging.error(f"Error cleaning up temp file {pdf_path}: {str(e)}")
        
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Error cleaning up temp directory: {str(e)}")
        
        # Schedule cleanup task
        background_tasks.add_task(cleanup_temp_files)