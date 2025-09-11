import os
import json
from io import BytesIO
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
import requests

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

    def load_from_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Load product data from a URL (CSV or JSON).

        Args:
            url: HTTP/HTTPS URL to a CSV or JSON feed

        Returns:
            List of product dicts
        """
        logger.info(f"Fetching catalog from URL: {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch URL: {e}")
            raise

        content_type = resp.headers.get("Content-Type", "").lower()
        data: List[Dict[str, Any]]

        try:
            if "json" in content_type or url.lower().endswith((".json",)):
                payload = resp.json()
                # Expect either a list of products or an object with a key holding the list
                if isinstance(payload, list):
                    data = payload
                elif isinstance(payload, dict):
                    # Try common keys
                    for key in ("products", "items", "data", "catalog"):
                        if key in payload and isinstance(payload[key], list):
                            data = payload[key]
                            break
                    else:
                        # Fallback: wrap the dict as single item
                        data = [payload]
                else:
                    raise ValueError("Unsupported JSON structure for catalog")
            else:
                # Assume CSV by default
                df = pd.read_csv(BytesIO(resp.content))
                data = df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to parse feed: {e}")
            raise

        # Normalize fields to engine schema
        normalized = [self._normalize_product_record(rec) for rec in data]
        return normalized

    def _normalize_product_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Map common ecommerce fields to the expected schema.

        Expected keys:
          product_id, name, category, price, description, features, in_stock, rating
        """
        def pick(keys, default=None):
            for k in keys:
                if k in rec and rec[k] is not None and rec[k] != "":
                    return rec[k]
            return default

        product_id = pick(["product_id", "id", "sku", "uid"]) or str(hash(json.dumps(rec, sort_keys=True)))
        name = pick(["name", "title", "product_name"]) or "Unnamed Product"
        category = pick(["category", "type", "collection"]) or "General"
        price_val = pick(["price", "amount", "sale_price", "regular_price"], 0)
        try:
            price = float(price_val)
        except Exception:
            price = 0.0
        description = pick(["description", "desc", "details", "summary"]) or ""
        features = pick(["features", "specs", "specifications"], "")
        in_stock_raw = pick(["in_stock", "stock", "availability", "available"], True)
        in_stock = bool(in_stock_raw in [True, 1, "1", "true", "True", "in_stock", "available", "yes", "Yes"]) if not isinstance(in_stock_raw, bool) else in_stock_raw
        rating_val = pick(["rating", "rating_value", "stars"], 0)
        try:
            rating = float(rating_val)
        except Exception:
            rating = 0.0

        return {
            "product_id": product_id,
            "name": name,
            "category": category,
            "price": price,
            "description": description,
            "features": features,
            "in_stock": in_stock,
            "rating": rating,
        }
    
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
    
    def ingest_and_index(self, file_path: Optional[str] = None, source_url: Optional[str] = None) -> bool:
        """
        Ingest product data from a file and create embeddings.
        
        Args:
            file_path: Path to the product data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load products from file or URL
            if source_url:
                products = self.load_from_url(source_url)
            elif file_path:
                products = self.load_from_csv(file_path)
            else:
                raise ValueError("Either file_path or source_url must be provided")
            
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
