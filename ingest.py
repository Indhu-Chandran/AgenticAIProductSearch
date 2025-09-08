import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI()

class ProductCatalogIngester:
    """
    Class for ingesting product catalog data and creating embeddings.
    Supports CSV files and can be extended to support other formats.
    """
    
    def __init__(self, persist_dir: str = "./storage"):
        """
        Initialize the ingester with a directory to persist the data.
        
        Args:
            persist_dir: Directory to persist the data
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.products_file = os.path.join(persist_dir, "products.json")
        self.embeddings_file = os.path.join(persist_dir, "embeddings.json")
        
    def load_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load product data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of product dictionaries
        """
        logger.info(f"Loading product data from {file_path}")
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    
    def create_product_embeddings(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create embeddings for each product using OpenAI's embedding API.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Dictionary mapping product IDs to embeddings
        """
        embeddings = {}
        
        for product in products:
            # Create a structured text representation of the product
            content = f"""Product ID: {product['product_id']}\n
            Name: {product['name']}\n
            Category: {product['category']}\n
            Price: ${product['price']}\n
            Description: {product['description']}\n
            Features: {product['features']}\n
            In Stock: {'Yes' if product['in_stock'] else 'No'}\n
            Rating: {product['rating']} / 5.0"""
            
            try:
                # Get embedding from OpenAI using the modern client
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=content
                )
                
                # Store embedding with product ID as key
                product_id = str(product['product_id'])
                embeddings[product_id] = response.data[0].embedding
                
            except Exception as e:
                logger.error(f"Error creating embedding for product {product['product_id']}: {e}")
        
        logger.info(f"Created embeddings for {len(embeddings)} products")
        return embeddings
    
    def ingest_and_index(self, file_path: str) -> bool:
        """
        Ingest product data from a file and create embeddings.
        
        Args:
            file_path: Path to the product data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load products from file
            products = self.load_from_csv(file_path)
            
            # Create embeddings
            embeddings = self.create_product_embeddings(products)
            
            # Save products and embeddings
            with open(self.products_file, 'w') as f:
                json.dump(products, f)
                
            with open(self.embeddings_file, 'w') as f:
                # Convert embeddings to list for JSON serialization
                serializable_embeddings = {k: list(v) for k, v in embeddings.items()}
                json.dump(serializable_embeddings, f)
            
            logger.info(f"Saved products and embeddings to {self.persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            return False
    
    def load_data(self) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, List[float]]]]:
        """
        Load previously persisted products and embeddings.
        
        Returns:
            Tuple of (products, embeddings) if they exist, (None, None) otherwise
        """
        if not (os.path.exists(self.products_file) and os.path.exists(self.embeddings_file)):
            logger.warning(f"No data found at {self.persist_dir}")
            return None, None
        
        try:
            with open(self.products_file, 'r') as f:
                products = json.load(f)
                
            with open(self.embeddings_file, 'r') as f:
                embeddings = json.load(f)
                
            logger.info(f"Loaded {len(products)} products and {len(embeddings)} embeddings")
            return products, embeddings
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None
