import streamlit as st
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ingest import ProductCatalogIngester
from agent import ProductSearchAgent
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set it in a .env file.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Product Catalog Query Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def ingest_catalog(file_path):
    """Ingest product catalog and create agent"""
    with st.spinner("Ingesting product catalog..."):
        ingester = ProductCatalogIngester(persist_dir="./storage")
        success = ingester.ingest_and_index(file_path)
        
        if success:
            # Load the products and embeddings
            products, embeddings = ingester.load_data()
            if products and embeddings:
                st.session_state.agent = ProductSearchAgent(products, embeddings)
                st.session_state.ingested = True
                st.success("Product catalog ingested successfully!")
            else:
                st.error("Failed to load products and embeddings after ingestion.")
        else:
            st.error("Failed to ingest product catalog.")

def load_existing_data():
    """Load existing data if available"""
    with st.spinner("Loading existing data..."):
        ingester = ProductCatalogIngester(persist_dir="./storage")
        products, embeddings = ingester.load_data()
        
        if products and embeddings:
            st.session_state.agent = ProductSearchAgent(products, embeddings)
            st.session_state.ingested = True
            st.success("Loaded existing data successfully!")
            return True
        return False

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    # Header
    st.title("🔍 Product Catalog Query Engine")
    st.markdown("""
    This demo showcases an Agentic AI application using OpenAI embeddings to create a powerful 
    query engine for product catalogs. Ask questions about products in natural language!
    """)
    
    # Auto-ingest from environment URL (if provided and nothing loaded yet)
    catalog_url = os.getenv("CATALOG_URL")
    if not st.session_state.ingested and catalog_url:
        try:
            with st.spinner("Auto-ingesting catalog from CATALOG_URL..."):
                ingester = ProductCatalogIngester(persist_dir="./storage")
                if ingester.ingest_and_index(source_url=catalog_url):
                    products, embeddings = ingester.load_data()
                    if products and embeddings:
                        st.session_state.agent = ProductSearchAgent(products, embeddings)
                        st.session_state.ingested = True
                        st.success("Catalog auto-ingested from CATALOG_URL")
        except Exception as e:
            st.warning(f"Auto-ingest failed: {e}")

    # Sidebar
    with st.sidebar:
        st.header("Product Catalog")
        
        # Option to use sample data, upload file, or load from URL
        data_option = st.radio(
            "Choose data source:",
            ["Use sample catalog", "Upload custom catalog", "Load from URL"]
        )
        
        if data_option == "Use sample catalog":
            if st.button("Load Sample Catalog"):
                ingest_catalog("data/product_catalog.csv")
        elif data_option == "Upload custom catalog":
            uploaded_file = st.file_uploader("Upload your product catalog (CSV)", type=["csv"])
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("data/uploaded_catalog.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                ingest_catalog("data/uploaded_catalog.csv")
        else:
            url = st.text_input("Enter catalog URL (CSV or JSON)", placeholder="https://example.com/products.json")
            if st.button("Load from URL") and url:
                with st.spinner("Fetching and ingesting catalog from URL..."):
                    ingester = ProductCatalogIngester(persist_dir="./storage")
                    ok = ingester.ingest_and_index(source_url=url)
                    if ok:
                        products, embeddings = ingester.load_data()
                        if products and embeddings:
                            st.session_state.agent = ProductSearchAgent(products, embeddings)
                            st.session_state.ingested = True
                            st.success("Catalog ingested from URL successfully!")
                        else:
                            st.error("Fetched, but failed to load products/embeddings after ingestion.")
                    else:
                        st.error("Failed to ingest from the provided URL. Please verify the URL returns CSV or JSON.")
        
        # Option to load existing data
        if not st.session_state.ingested:
            storage_dir = "./storage"
            products_file = os.path.join(storage_dir, "products.json")
            embeddings_file = os.path.join(storage_dir, "embeddings.json")
            
            if os.path.exists(products_file) and os.path.exists(embeddings_file):
                if st.button("Load Existing Data"):
                    load_existing_data()
        
        # Display catalog preview if ingested
        if st.session_state.ingested and hasattr(st.session_state.agent, 'products'):
            st.subheader("Catalog Preview")
            try:
                df = pd.DataFrame(st.session_state.agent.products)
                st.dataframe(df[["name", "category", "price"]].head())
            except Exception as e:
                st.error(f"Error loading catalog preview: {e}")
        
        # Reset chat
        if st.session_state.ingested and len(st.session_state.chat_history) > 0:
            if st.button("Reset Chat"):
                st.session_state.chat_history = []
                if st.session_state.agent:
                    st.session_state.agent.reset()
                st.experimental_rerun()
    
    # Main content area - Chat interface
    if st.session_state.ingested and st.session_state.agent:
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_query = st.chat_input("Ask about products...")
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.chat(user_query)
                st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        # Prompt to ingest data
        st.info("Please ingest a product catalog using the sidebar options to start.")
        
        # Example queries
        st.subheader("Example queries you can ask:")
        examples = [
            "What electronics products do you have?",
            "Show me kitchen appliances under $150",
            "I need a new smartphone with good camera quality",
            "Compare the Ultra HD Smart TV with the Smartphone XS Pro",
            "What's your highest rated product in Electronics?",
            "Recommend me a product for my home office setup"
        ]
        for example in examples:
            st.markdown(f"- *{example}*")

if __name__ == "__main__":
    main()
