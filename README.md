# Multi-modal RAG Based Application for CFA Publications
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)](https://www.snowflake.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Milvus](https://img.shields.io/badge/Milvus-00A4E4?style=for-the-badge&logo=milvus&logoColor=white)](https://milvus.io/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

An interactive RAG based application built using FastAPI and Streamlit to explore and analyze publications from the CFA Institute Research Foundation. The application extract contents from the publications including images, graphs, PDF files and stores them in Snowflake database and Chroma DB. Users can interactively explore documents, generate on-the-fly summaries, and retrieve insights using multi-modal Retrieval-Augmented Generation (RAG) approach. This application supports Q/A functionality, incremental indexing of research notes, and comprehensive search within documents enhancing document discovery and analysis.


### Problem Statement
The challenge we addressed was to create an intelligent system that could effectively process, analyze, and present CFA Institute's research publications. We needed to handle both text and visual content, implement retrieval mechanisms, and provide users with tools to interact with this knowledge base effectively.

With the rapid growth of publications, it has become increasingly challenging to analyze complex documents, interpret images and graphs, and derive meaningful business insights. This project aims to create a robust solution that allows users to interact with documents through Q&A functionality. By leveraging Retrieval-Augmented Generation (RAG), the application provides relevant responses, images, and graphs specific to user questions, simplifying the analysis process and enabling users to obtain accurate insights efficiently. The application primarily focuses on:

Content Extraction and Storage: Extract content from the CFA Institute Research Foundation website, loading files onto S3 and storing textual data, such as titles and summaries, in a Snowflake database.
Automated Data Ingestion: Automate the data ingestion process with an Airflow pipeline, ensuring efficient and structured data handling.
API Endpoints: Develop multiple API endpoints to support services like document exploration and dropdown selection for document access, RAG interaction with UI.
Real-Time Summary Generation: Generate document summaries on the fly using NVIDIA’s advanced services.
Multi-Modal RAG with Cleanlabs Integration: Implement multi-modal RAG to enhance response relevance and integrate Cleanlabs for response trustworthiness.
Comprehensive Report Generation: Create reports that include responses, images, and graphs relevant to user queries for a richer understanding.
Research Notes Validation and Indexing: Validate, store, and incrementally index research notes to facilitate efficient future searches and analysis.

### Scope
The project encompasses several key areas that work together to create a comprehensive solution:

The Data Ingestion Pipeline utilizes Apache Airflow to orchestrate the scraping and processing of CFA publications. This system automatically extracts text and images from documents, ensuring all content is properly captured and organized for further processing.

Our Storage Solution combines AWS S3 for document storage with Snowflake for metadata management. This dual-storage approach ensures that both structured and unstructured data are handled efficiently while maintaining quick access capabilities.

The Multi-modal RAG Implementation represents the core of our system. It processes both text and images intelligently, understanding the relationships between different content types and enabling sophisticated query responses. This system ensures that users receive contextually relevant information from their queries.

The User Interface provides an intuitive way to interact with the system. Through Streamlit, we've created a responsive frontend that allows users to explore documents, ask questions, and generate comprehensive reports. The interface is designed to be both powerful and easy to use.

The Backend Services, built with FastAPI, handle all the complex processing while ensuring security and performance. These services manage everything from user authentication to document analysis, providing a robust foundation for the entire system.

### Key Features

Our Data Processing capabilities are comprehensive and automated. The system handles document extraction, including both text and images, and processes this content to enable intelligent retrieval. The processed data is then stored securely and efficiently in snowflake and AWS.

The Document Analysis system is powered by llm models that understand both text and visual content. Users can generate summaries, ask questions, and receive answers that draw from the entire knowledge base. The system maintains context awareness, ensuring responses are relevant and accurate.

Our Report Generation feature creates professional documents that combine text and visual elements in the responses. These visual knowledge in reports make it easy to share insights from the analysis.

## Architecture Diagram:
![multi-modal_rag_system](https://github.com/user-attachments/assets/0efea28d-9969-48b4-8fb9-2073f7508273)

## Detailed System Overview

### 1. Data Acquisition and Processing
The system begins with automated data collection from CFA Institute Research Foundation Publications:

- **Web Scraping (Airflow DAG)**:
  - Extracts PDF files, titles, and summaries
  - Downloads images from publications
  - Processes metadata systematically

- **Storage Integration**:
  - AWS S3 buckets for PDFs and images
  - Snowflake database for structured metadata

### 2. Multi-modal RAG Implementation
Advanced RAG system combining text and image analysis:

- **Text Processing**:
  - Document chunking and indexing
  - Semantic understanding
  - Vector embeddings generation

- **Image Processing**:
  - Visual content extraction
  - Image-text alignment
  - Multi-modal context integration

- **RAG Components**:
  - OpenAI API/ NVIDIA with llama utilization
  - Vector DB management

### 3. Security and Authentication
Robust security measures protect user data and system access:

- **User Authentication**:
  - JWT-based authentication
  - Secure password handling
  - Session management

### 4. User Interface Features
Comprehensive interface for document interaction:

- **Document Selection**:
  - Grid view with thumbnails, titles

- **Document Viewer**:
  - PDF rendering
  - Image display
  - Text extraction view
  - Navigation controls

- **Analysis Tools**:
  - Interactive Q&A interface
  - Summary generation
  - Report creation

## Technical Details

### Data Flow
1. **Data Ingestion**

2. **Document Processing**

3. **RAG Implementation**


## Deployment
Containerization of FastAPI and Streamlit applications using Docker
Deployment to a public cloud platform using Docker Compose
Ensuring public accessibility of the deployed applications - Streamlit and FastAPI
Providing clear instructions for users to interact with the RAG application and explore its functionalities
The FastAPI and Streamlit are containerized using Docker, and orchestrated through docker compose and the Docker images are pushed to Docker Hub. For deploying the Docker containers, we use an Google Cloud Platform (GCP)


### Data Source
CFA Institute Research Foundation Publications: https://rpc.cfainstitute.org/en/research-foundation/publications#sort=%40officialz32xdate%20descending&f:SeriesContent=%5BResearch%20Foundation%5D

### Screenshots 

Landing Page
![image](https://github.com/user-attachments/assets/b8f77f0e-6bd3-44d7-abf5-f4628cfa483c)

Dashboard Page
![image](https://github.com/user-attachments/assets/7a273217-67a1-4bd6-a973-6f3f807758b3

Document Analysis Page
![image](https://github.com/user-attachments/assets/38b586dc-9527-45d8-bf1a-2931285866ca)

Chat with Pdf Page
![image](https://github.com/user-attachments/assets/6de65f3d-669f-4817-a295-796b8b81b96b)


## Setup and Deployment

### Prerequisites
- Docker and Docker Compose v2.0+
- AWS Account with S3 access
- Snowflake Account with admin privileges
- OpenAI API Key
- NVIDIA API Key (optional)

## Set Up Application Locally
1. Clone the repository to get all the source code on your machine 
```
git clone yourRepo
cd yourRepo
```
2. Set Up Environment Variables at required locations by creating .env files with variable values.
```
#env at Airflow

GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/"your GCP Service Account json"
AIRFLOW_UID=502

```

3. Build and Run Airflow by running below commands
```
cd Airflow
docker build -t my_airflow_image:latest .
docker-compose up -d
```
Access the Airflow web UI at http://localhost:8080

4. Trigger the Airflow DAG in the Airflow UI to start the data pipeline.
5. Navigate back to root dir to setup relevant env variables for streamlit and FastAPI applications
```
cd ..
```

```
#env for root dir

DB_HOST="your DB Host"
DB_PORT=5432 # default
DB_NAME=""your DB name
DB_USER="your DB user"
DB_PASSWORD="your DB pwd"

API_URL=http://fastapi:8000

SECRET_KEY="your secret key"
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
GOOGLE_APPLICATION_CREDENTIALS="your GCP Service Account json"
OPENAI_API_KEY="your openai API Key"
NVIDIA_API_KEY="your key"

SNOWFLAKE_ACCOUNT="your details"
SNOWFLAKE_USER="your details"
SNOWFLAKE_PASSWORD="your details"
SNOWFLAKE_DATABASE="your details"
SNOWFLAKE_SCHEMA="your details"
SNOWFLAKE_WAREHOUSE="your details"
SNOWFLAKE_TABLE="your details"

AWS_ACCESS_KEY_ID="your details"
AWS_SECRET_ACCESS_KEY="your details"
AWS_REGION=us-east-1
BUCKET_NAME="your details"
PDF_FOLDER="your details"
IMAGE_FOLDER="your details"

MILVUS_HOST=localhost
MILVUS_PORT=19530

LLAMA_PARSE_API_KEY="your key"
```

6. Local docker compose build and up, push the images to hub
 + build fastapi and streamlit docker images through docker compose from root dir
 ```
 docker compose build --no-cache
 ```
 + Runs the images thorugh docker compose
 ```
 docker compose up
 ```
 + Tag the FastAPI image:
 ```
 docker tag ImageNameForFastapi Username/ImageNameForFastapi:latest
 ```
 + Tag the Streamlit image:
 ```
 docker tag ImageNameForStreamlit Username/ImageNameForStreamlit:latest
 ```
 + Push FastAPI:
 ```
 docker push Username/ImageNameForFastapi:latest
 ```
 + Push Streamlit:
 ```
 docker push Username/ImageNameForStreamlit:latest
 ```
 
## Deploy in GCP VM

1. GCP docker setup, create folder, create docker compose file, scp the .env and json file, pull the images, run docker compose.
 + Install Docker:
 ```
 sudo apt update
 sudo apt install -y docker.io
 ```
 + Install Docker Compose:
 ```
 sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
 sudo chmod +x /usr/local/bin/docker-compose
 ```
 + Create a Directory for Your Project:
 ```
 mkdir ~/yourapp
 cd ~/yourapp
 ```
 + scp json file to myapp and .env
 ```
 gcloud compute scp --project YourProjectName --zone YourZone-f /path to your root dir/ServiceAccountJson Username@InstanceName:/PathInGCPToyourapp/ServiceAccountJson
 gcloud compute scp --project YourProject --zone YourZone-f /path to your root dir/.env Username@InstanceName:/PathInGCPToyourapp/.env
 ```
2. nano docker-compose.yml:
```
services:
  fastapi:
    image: Username/ImageNameForFastapi:latest  # Pull from Docker Hub
    container_name: YourFastapiContainer
    ports:
      - "8000:8000"
    env_file:
      - ./.env  # Pass the .env file located in the root directory at runtime
    volumes:
      - ./ServiceAccountJson:/app/ServiceAccountJson  # Mount the JSON file at runtime
    networks:
      - app-network

  streamlit:
    image: Username/ImageNameForStreamlit:latest  # Pull from Docker Hub
    container_name: YourStreamlitContainer
    ports:
      - "8501:8501"
    depends_on:
      - fastapi
    env_file:
      - ./.env  # Pass the .env file located in the root directory at runtime
    volumes:
      - ./ServiceAccountJson:/app/ServiceAccountJson  # Mount the JSON file at runtime
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

3. Pull the Docker images and Start the containers:
```
sudo docker-compose pull
sudo docker-compose up -d
```






