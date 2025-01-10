import streamlit as st
import requests
import os
from typing import List
from dotenv import load_dotenv
import base64
from io import BytesIO
import markdown

# Load environment variables
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'document_select'
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_question' not in st.session_state:
    st.session_state.processing_question = False

# Custom CSS styles
def load_css():
    st.markdown("""
        <style>
        /* Existing styles... */
        
        /* Document card styling */
        .document-card {
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
            background: white;
            transition: all 0.3s ease;
        }
        .document-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Document viewer styling */
        .document-viewer {
            height: 500px;
            overflow-y: scroll;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
            margin-bottom: 20px;
        }
        
        /* Analysis options styling */
        .analysis-option {
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .analysis-option:hover {
            background: #e9ecef;
        }
        </style>
    """, unsafe_allow_html=True)

def initial_home():
    # Set page layout
    st.set_page_config(page_title="Multi-modal RAG", page_icon="📄", layout="wide")
    load_css()

    # Create a header container
    header_container = st.container()
    with header_container:
        # Create columns for title and login button with better ratio
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col2:
            st.markdown(
                '<div style="text-align: center; font-size: 3.5rem; font-weight: bold;">Multi-modal RAG</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            # Add vertical margin to align with title
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            if st.session_state.logged_in:
                if st.button("🚪 Logout", key="logout_button", use_container_width=True):
                    # Handle logout
                    st.session_state.logged_in = False
                    st.session_state.token = None
                    st.session_state.username = None
                    st.session_state.page = "home"
                    st.session_state.chat_history = []
                    st.session_state.documents = []
                    st.session_state.selected_documents = []
                    st.session_state.current_page = 'document_select'
                    st.session_state.current_document = None
                    st.session_state.processing_question = False
                    st.rerun()
            else:
                if st.button("🔒 Login", key="login_button", use_container_width=True):
                    st.session_state.page = "signin"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    if st.session_state.logged_in:
        st.markdown(
            f'<div style="text-align: center; font-size: 2.5rem; margin-top: 2rem;">Welcome, {st.session_state.username}!</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem;">You are now logged in to Multi-modal RAG.</div>',
            unsafe_allow_html=True)
    else:
        # Landing page content
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem; margin-top: 2rem;">Your AI-powered assistant for Multi-modal Retrieval Augmented Generation</div>',
            unsafe_allow_html=True
        )
        
        st.write("---")
        
        # Features section
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem;">✨ Features</div>',
            unsafe_allow_html=True)
        st.markdown('''
            <div style="text-align: center;">
            <ul class="features">
                <li>🔍 <strong>Retrieve</strong> information from various modalities like text, images.</li>
                <li>🤖 <strong>Generate</strong> responses augmented with retrieved data.</li>
                <li>📈 <strong>Improve</strong> AI performance with context-aware retrieval.</li>
            </ul>
            </div>
        ''', unsafe_allow_html=True)
        
        st.write("---")
        
        # Get Started section
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem;">🚀 Get Started</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align:center;">Click the <strong>🔒 Login</strong> button at the top right to sign in or create a new account.</p>',
            unsafe_allow_html=True)
        
        # Footer
        st.markdown(
            '<div class="footer" style="text-align: center; margin-top: 2rem;">© 2024 Multi-modal RAG. All rights reserved.</div>',
            unsafe_allow_html=True)
        
def add_message(role: str, content: str, sources: List = None):
    """Add a message to the chat history"""
    message = {
        "role": role,
        "content": content
    }
    if sources:
        message["sources"] = sources
    st.session_state.chat_history.append(message)

def clear_chat():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.processing_question = False

def handle_api_error(error: Exception) -> str:
    """Format API errors for display"""
    if isinstance(error, requests.exceptions.RequestException):
        if hasattr(error, 'response') and error.response is not None:
            try:
                error_data = error.response.json()
                return f"API Error: {error_data.get('detail', str(error))}"
            except ValueError:
                return f"API Error: {str(error)}"
        return f"Connection Error: {str(error)}"
    return f"Error: {str(error)}"

# 3. API CALL WRAPPER - After utility functions
def api_call(func):
    """Decorator to handle API calls and errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = handle_api_error(e)
            st.error(error_message)
            return None
    return wrapper
        
def register():
    st.set_page_config(page_title="Sign Up", page_icon="🔑")
    load_css()

    # Header with back button
    col1, col2 = st.columns([8, 1.5])
    with col1:
        st.markdown(
            '<div style="text-align: left; font-size: 3rem; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;">🔑 Sign Up</div>',
            unsafe_allow_html=True)
    with col2:
        st.button("🔙 Back", on_click=lambda: st.session_state.update(page="signin"))

    # Registration form
    st.write("Please fill in the details below to create a new account.")
    email = st.text_input("📧 Email", key="signup_email")
    username = st.text_input("👤 Username", key="signup_username")
    password = st.text_input("🔒 Password", type="password", key="signup_password")
    password_confirm = st.text_input("🔒 Confirm Password", type="password", key="signup_password_confirm")

    def handle_register():
        if not email or not username or not password or not password_confirm:
            st.warning("⚠️ Please fill in all fields")
        elif password != password_confirm:
            st.warning("⚠️ Passwords do not match")
        else:
            response = requests.post(
                f"{API_URL}/register",
                json={
                    "email": email,
                    "username": username,
                    "password": password
                }
            )
            if response.status_code == 200:
                st.success("🎉 Registration successful! Please sign in.")
                # Clear the input fields
                st.session_state.signup_email = ''
                st.session_state.signup_username = ''
                st.session_state.signup_password = ''
                st.session_state.signup_password_confirm = ''
                st.session_state.page = "signin"
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Registration failed: {error_detail}")

    st.button("✅ Register", on_click=handle_register)

def login():
    st.set_page_config(page_title="Sign In", page_icon="🔐")
    load_css()

    # Header with back button
    col1, col2 = st.columns([8, 1.5])
    with col1:
        st.markdown(
            '<div style="text-align: left; font-size: 3rem; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;">🔐 Sign In</div>',
            unsafe_allow_html=True)
    with col2:
        st.button("🔙 Back", on_click=lambda: st.session_state.update(page="home"))

    st.write("Please enter your credentials to sign in.")

    username = st.text_input("👤 Username", key="signin_username")
    password = st.text_input("🔒 Password", type="password", key="signin_password")

    def handle_login():
        if not username or not password:
            st.warning("⚠️ Please enter both username and password")
        else:
            response = requests.post(
                f"{API_URL}/login",
                data={
                    "username": username,
                    "password": password
                }
            )
            if response.status_code == 200:
                token = response.json().get("access_token")
                st.session_state.token = token
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.signin_username = ''
                st.session_state.signin_password = ''
                st.session_state.current_page = 'document_select'  # Changed from 'home'
                st.session_state.page = "home"
                st.success("✅ Logged in successfully!")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Login failed: {error_detail}")

    def go_to_signup():
        st.session_state.signin_username = ''
        st.session_state.signin_password = ''
        st.session_state.page = "signup"

    st.button("➡️ Sign In", on_click=handle_login)
    st.write("Don't have an account?")
    st.button("📝 Sign Up", on_click=go_to_signup)

def chat_interface():
    """Updated chat interface with immediate responses and visible logout"""
    # Header with title and logout
    header_col1, header_col2 = st.columns([10, 2])
    
    with header_col1:
        st.title("💬 Multi-Document Q&A")
    
    with header_col2:
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.page = "home"
            st.session_state.chat_history = []
            st.session_state.documents = []
            st.session_state.selected_documents = []
            st.session_state.current_page = 'document_select'
            st.session_state.current_document = None
            st.session_state.processing_question = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show which documents are being analyzed
    st.write("📚 Analyzing the following documents:")
    for doc in st.session_state.selected_documents:
        st.markdown(f"- **{doc['title']}**")
    
    # Create a container for chat display
    chat_container = st.container()
    
    # Display chat history in the container
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    st.markdown("#### Sources Used:")
                    for source in message["sources"]:
                        st.markdown(f"""
                        - **Document**: {source['title']}
                        - **Source**: {source['source']}
                        - **Relevance Score**: {source['relevance_score']:.2f}
                        """)

    # Question input
    if question := st.chat_input("Ask a question about the documents..."):
        add_message("user", question)
        
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing documents..."):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    response = requests.post(
                        f"{API_URL}/analyze-documents",
                        headers=headers,
                        json={
                            "pdf_links": [doc['pdf_link'] for doc in st.session_state.selected_documents],
                            "analysis_type": "qa",
                            "question": question
                        }
                    )
                    
                    if response.status_code == 200:
                        answer_data = response.json()
                        message_placeholder.markdown(answer_data["answer"])
                        add_message(
                            "assistant",
                            answer_data["answer"],
                            answer_data.get("source_documents", [])
                        )
                        
                        if answer_data.get("source_documents"):
                            st.markdown("#### Sources Used:")
                            for source in answer_data["source_documents"]:
                                st.markdown(f"""
                                - **Document**: {source['title']}
                                - **Source**: {source['source']}
                                - **Relevance Score**: {source['relevance_score']:.2f}
                                """)
                    else:
                        error_message = f"Error: {response.json().get('detail', 'Unknown error')}"
                        message_placeholder.error(error_message)
                        add_message("assistant", error_message)
                except Exception as e:
                    error_message = f"Error communicating with API: {str(e)}"
                    message_placeholder.error(error_message)
                    add_message("assistant", error_message)

    # Navigation at the bottom
    st.markdown("---")
    if st.button("⬅️ Back to Documents", use_container_width=True):
        st.session_state.current_page = 'document_view'
        st.rerun()

def fetch_documents():
    """Fetch available documents from API"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{API_URL}/documents", headers=headers)
        if response.status_code == 200:
            st.session_state.documents = response.json()
            return True
        return False
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return False

# Add this to your session state initialization at the top
if 'document_content_cache' not in st.session_state:
    st.session_state.document_content_cache = {}

def document_selection_view():
    # Create a container for the header section
    header_container = st.container()
    
    with header_container:
        # Create columns for title, next button, and logout button
        title_col, next_col, logout_col = st.columns([7, 2, 1])
        with title_col:
            st.title("📚 Document Selection")
        with next_col:
            # Add vertical margin to align with title
            st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
            if st.button("Next ➡️", key="top_next", use_container_width=True):
                st.session_state.current_page = 'document_view'
                # Clear current document and cache when moving to next page
                st.session_state.current_document = None
                st.session_state.document_content_cache = {}
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with logout_col:
            st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
            if st.button("🚪", key="logout_btn", help="Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.token = None
                st.session_state.username = None
                st.session_state.page = "home"
                st.session_state.chat_history = []
                st.session_state.documents = []
                st.session_state.selected_documents = []
                st.session_state.current_page = 'document_select'
                st.session_state.current_document = None
                st.session_state.processing_question = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Rest of the document selection code...
    if not st.session_state.documents:
        with st.spinner("Loading documents..."):
            fetch_documents()
    
    if st.session_state.documents:
        st.write("Select one or more documents to analyze:")
        
        # Create a grid of documents
        cols = st.columns(3)
        for idx, doc in enumerate(st.session_state.documents):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px;'>
                    <img src="{doc['image_link']}" style="width:100%; border-radius:5px;">
                    <h4>{doc['title']}</h4>
                    <p>{doc['brief_summary'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Checkbox for selection
                is_selected = st.checkbox(
                    "Select",
                    key=f"select_{idx}",
                    value=doc['pdf_link'] in [d['pdf_link'] for d in st.session_state.selected_documents]
                )
                
                if is_selected and doc not in st.session_state.selected_documents:
                    st.session_state.selected_documents.append(doc)
                elif not is_selected and doc in st.session_state.selected_documents:
                    st.session_state.selected_documents.remove(doc)

def document_viewer():
    # First create the header with navigation
    header_col1, header_col2 = st.columns([10, 2])
    
    with header_col1:
        st.title("📄 Document Analysis")
    
    with header_col2:
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.page = "home"
            st.session_state.chat_history = []
            st.session_state.documents = []
            st.session_state.selected_documents = []
            st.session_state.current_page = 'document_select'
            st.session_state.current_document = None
            st.session_state.processing_question = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Selected Documents")
        # Display selected documents as clickable cards
        for doc in st.session_state.selected_documents:
            with st.container():
                st.markdown(f"""
                <div style='padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin: 5px; cursor: pointer;'>
                    <h4>{doc['title']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View Document", key=f"view_{doc['title']}"):
                    st.session_state.current_document = doc
                    # Clear cache for this document to force reload
                    if doc['pdf_link'] in st.session_state.document_content_cache:
                        del st.session_state.document_content_cache[doc['pdf_link']]
                    st.rerun()
    
    with col2:
        if st.session_state.current_document:
            st.subheader(f"📑 {st.session_state.current_document['title']}")
            
            # Check if content is in cache
            pdf_link = st.session_state.current_document['pdf_link']
            if pdf_link not in st.session_state.document_content_cache:
                # Fetch and cache document content
                with st.spinner("Loading document..."):
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        response = requests.get(
                            f"{API_URL}/document-content",
                            headers=headers,
                            params={"pdf_link": pdf_link},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            content_data = response.json()
                            st.session_state.document_content_cache[pdf_link] = content_data['content']
                        else:
                            st.error(f"Error loading document: {response.json().get('detail', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            # Display cached content
            if pdf_link in st.session_state.document_content_cache:
                st.markdown(
                    f"""<div style='height: 400px; overflow-y: scroll; padding: 20px; 
                    border: 1px solid #ddd; border-radius: 5px; background-color: white;'>
                    {st.session_state.document_content_cache[pdf_link]}
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.info("👈 Select a document from the left to view its content")

    # Add unified analysis section below the document viewer
    st.markdown("---")
    st.subheader("🔍 Document Analysis")

    # Analysis options in three columns
    col_sum, col_qa, col_report = st.columns(3)

    with col_sum:
        if st.button("📝 Generate Combined Summary", use_container_width=True):
            if not st.session_state.selected_documents:
                st.warning("Please select at least one document.")
            else:
                progress_text = "Generating comprehensive summary..."
                with st.spinner(progress_text):
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        response = requests.post(
                            f"{API_URL}/analyze-documents",
                            headers=headers,
                            json={
                                "pdf_links": [doc['pdf_link'] for doc in st.session_state.selected_documents],
                                "analysis_type": "summary"
                            },
                            timeout=60  # Increased timeout for larger documents
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Create an expander for the summary
                            with st.expander("📋 Combined Summary", expanded=True):
                                st.write(result["answer"])
                                
                                # Display sources if available
                                if result.get("source_documents"):
                                    st.markdown("#### Sources Used:")
                                    for source in result["source_documents"]:
                                        st.markdown(f"""
                                        - **Document**: {source['title']}
                                        - **Relevance Score**: {source['relevance_score']:.2f}
                                        """)
                        else:
                            st.error(f"Error generating summary: {response.json().get('detail', 'Unknown error')}")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The documents might be too large. Try selecting fewer documents.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_qa:
        if st.button("❓ Q&A Mode", use_container_width=True):
            if not st.session_state.selected_documents:
                st.warning("Please select at least one document.")
            else:
                st.session_state.current_document = None
                st.session_state.document_content_cache = {}
                st.session_state.chat_history = []
                st.session_state.processing_question = False
                st.session_state.current_page = 'chat'
                st.rerun()

    with col_report:
        if st.button("📰 Generate Report", use_container_width=True):
            if not st.session_state.selected_documents:
                st.warning("Please select at least one document.")
            else:
                st.session_state.current_document = None
                st.session_state.document_content_cache = {}
                st.session_state.current_page = 'report_generation'
                st.rerun()

    # Navigation at the bottom
    st.markdown("---")
    if st.button("⬅️ Back to Selection", use_container_width=True):
        st.session_state.current_document = None
        st.session_state.document_content_cache = {}
        st.session_state.current_page = 'document_select'
        st.rerun()

def display_report(report_data):
    """Display report with text and images in Streamlit."""
    st.write("## 📄 Generated Report")
    
    if not report_data.get("blocks"):
        st.error("No content blocks found in the report")
        return
        
    for block in report_data["blocks"]:
        try:
            if isinstance(block, dict) and "text" in block:
                # Display text block
                st.markdown(block["text"])
                st.markdown("---")
            
            elif isinstance(block, dict) and "image_base64" in block:
                # Display image block
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(block["image_base64"])
                    
                    # Use BytesIO to create an in-memory bytes buffer
                    image_buffer = BytesIO(image_bytes)
                    
                    # Display image using st.image directly
                    if block.get("caption"):
                        st.image(image_buffer, caption=block["caption"], use_column_width=True)
                    else:
                        st.image(image_buffer, use_column_width=True)
                    
                    st.markdown("---")
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    st.exception(e)  # This will show the full traceback
            
        except Exception as e:
            st.error(f"Error displaying block: {str(e)}")
            continue

def report_generation_interface():
    # Header with title and logout
    header_col1, header_col2 = st.columns([10, 2])
    
    with header_col1:
        st.title("📊 Generate Multimodal Report")
    
    with header_col2:
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.page = "home"
            st.session_state.chat_history = []
            st.session_state.documents = []
            st.session_state.selected_documents = []
            st.session_state.current_page = 'document_select'
            st.session_state.current_document = None
            st.session_state.processing_question = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show selected documents
    st.write("📚 Generating report for the following documents:")
    for doc in st.session_state.selected_documents:
        st.markdown(f"- **{doc['title']}**")
    
    # Input for report query
    query = st.text_area(
        "Enter your report query:",
        height=100,
        help="Describe what kind of report you want to generate from the selected documents."
    )
    
    if st.button("Generate Report", type="primary"):
        if not query.strip():
            st.warning("Please enter a query for the report.")
            return
            
        try:
            with st.spinner("Generating report... This may take a few minutes"):
                # Setup request
                headers = {
                    "Authorization": f"Bearer {st.session_state.token}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    f"{API_URL}/generate-report",
                    headers=headers,
                    json={
                        "pdf_links": [doc['pdf_link'] for doc in st.session_state.selected_documents],
                        "question": query
                    },
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code == 200:
                    report_data = response.json()
                    
                    # Create expandable section for the report
                    with st.expander("📄 Generated Report", expanded=True):
                        display_report(report_data)
                        
                        # Add download button for the report
                        if report_data.get("blocks"):
                            # Convert report to HTML for download
                            html_content = convert_report_to_html(report_data)
                            
                            st.download_button(
                                label="Download Report",
                                data=html_content,
                                file_name="generated_report.html",
                                mime="text/html"
                            )
                else:
                    error_message = response.json().get('detail', 'Unknown error occurred')
                    st.error(f"Error generating report: {error_message}")
                    
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again with fewer documents or a simpler query.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # This will show the full traceback
    
    # Navigation
    st.markdown("---")
    if st.button("⬅️ Back to Document Viewer"):
        st.session_state.current_page = 'document_view'
        st.rerun()

def convert_report_to_html(report_data):
    """Convert report data to downloadable HTML format."""
    html_content = ["""
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            .text-block {
                margin: 1em 0;
            }
            .image-block {
                margin: 2em 0;
            }
            .image-block img {
                max-width: 100%;
                height: auto;
            }
            .caption {
                font-style: italic;
                color: #666;
                margin-top: 0.5em;
            }
        </style>
    </head>
    <body>
    """]
    
    for block in report_data["blocks"]:
        if "text" in block:
            # Convert markdown to HTML
            html_content.append(f"<div class='text-block'>{markdown.markdown(block['text'])}</div>")
        
        elif "image_base64" in block:
            # Embed base64 image
            caption = f"<div class='caption'>{block['caption']}</div>" if block.get('caption') else ""
            html_content.append(
                f"<div class='image-block'>"
                f"<img src='data:image/jpeg;base64,{block['image_base64']}' />"
                f"{caption}</div>"
            )
    
    html_content.append("</body></html>")
    return "\n".join(html_content)

def logged_in_view():
    if st.session_state.current_page == 'document_select':
        document_selection_view()
    elif st.session_state.current_page == 'document_view':
        document_viewer()
    elif st.session_state.current_page == 'chat':
        chat_interface()
    elif st.session_state.current_page == 'report_generation':
        report_generation_interface()

def home():
    if not st.session_state.logged_in:
        initial_home()
    else:
        logged_in_view()

def main():
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "signin":
        login()
    elif st.session_state.page == "signup":
        register()

if __name__ == "__main__":
    main()