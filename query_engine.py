import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI()

class ProductQueryEngine:
    """
    Query engine for product catalog data using OpenAI embeddings.
    """
    
    def __init__(self, products: List[Dict[str, Any]], embeddings: Dict[str, List[float]]):
        """
        Initialize the query engine with products and their embeddings.
        
        Args:
            products: List of product dictionaries
            embeddings: Dictionary mapping product IDs to embeddings
        """
        self.products = products
        self.embeddings = embeddings
        self.product_map = {str(p['product_id']): p for p in products}
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def _calculate_similarity(self, query_embedding: List[float], product_embedding: List[float]) -> float:
        """
        Calculate cosine similarity between query and product embeddings.
        
        Args:
            query_embedding: Query embedding vector
            product_embedding: Product embedding vector
            
        Returns:
            Cosine similarity score
        """
        query_array = np.array(query_embedding)
        product_array = np.array(product_embedding)
        
        # Normalize vectors
        query_norm = query_array / np.linalg.norm(query_array)
        product_norm = product_array / np.linalg.norm(product_array)
        
        # Calculate cosine similarity
        return float(np.dot(query_norm, product_norm))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for products based on a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of top matching products with similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Calculate similarity for each product
            results = []
            for product_id, product_embedding in self.embeddings.items():
                similarity = self._calculate_similarity(query_embedding, product_embedding)
                product = self.product_map.get(product_id)
                if product:
                    results.append({
                        "product": product,
                        "similarity": similarity
                    })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return top k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def filter_products(self, 
                       category: Optional[str] = None, 
                       min_price: Optional[float] = None,
                       max_price: Optional[float] = None,
                       in_stock_only: bool = False,
                       min_rating: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Filter products based on metadata.
        
        Args:
            category: Filter by product category
            min_price: Minimum price filter
            max_price: Maximum price filter
            in_stock_only: Filter for in-stock products only
            min_rating: Minimum rating filter
            
        Returns:
            List of filtered products
        """
        filtered_products = self.products.copy()
        
        # Apply filters
        if category:
            filtered_products = [p for p in filtered_products if p['category'] == category]
            
        if min_price is not None:
            filtered_products = [p for p in filtered_products if float(p['price']) >= min_price]
            
        if max_price is not None:
            filtered_products = [p for p in filtered_products if float(p['price']) <= max_price]
            
        if in_stock_only:
            filtered_products = [p for p in filtered_products if p['in_stock']]
            
        if min_rating is not None:
            filtered_products = [p for p in filtered_products if float(p['rating']) >= min_rating]
        
        return filtered_products
    
    def search_and_filter(self, 
                         query: str, 
                         category: Optional[str] = None,
                         min_price: Optional[float] = None,
                         max_price: Optional[float] = None,
                         in_stock_only: bool = False,
                         min_rating: Optional[float] = None,
                         top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search and filter products.
        
        Args:
            query: Search query
            category: Filter by product category
            min_price: Minimum price filter
            max_price: Maximum price filter
            in_stock_only: Filter for in-stock products only
            min_rating: Minimum rating filter
            top_k: Number of top results to return
            
        Returns:
            List of top matching filtered products with similarity scores
        """
        # First filter products
        filtered_products = self.filter_products(
            category=category,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            min_rating=min_rating
        )
        
        # Get filtered product IDs
        filtered_ids = [str(p['product_id']) for p in filtered_products]
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Calculate similarity for filtered products
        results = []
        for product_id in filtered_ids:
            if product_id in self.embeddings:
                product_embedding = self.embeddings[product_id]
                similarity = self._calculate_similarity(query_embedding, product_embedding)
                product = self.product_map.get(product_id)
                if product:
                    results.append({
                        "product": product,
                        "similarity": similarity
                    })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return results[:top_k]
