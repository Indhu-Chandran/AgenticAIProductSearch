import os
import json
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from query_engine import ProductQueryEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI()

class ProductSearchAgent:
    """
    Agentic AI for product search and recommendations using OpenAI.
    Fixed version that strictly returns only products from the catalog.
    """
    
    def __init__(self, products: List[Dict[str, Any]], embeddings: Dict[str, List[float]]):
        """
        Initialize the agent with products and their embeddings.
        
        Args:
            products: List of product dictionaries
            embeddings: Dictionary mapping product IDs to embeddings
        """
        self.products = products
        self.embeddings = embeddings
        self.query_engine = ProductQueryEngine(products, embeddings)
        self.chat_history = []
        
        # Create a quick lookup for categories and price ranges
        try:
            # Debug: Check the structure of products
            if products and len(products) > 0:
                logger.info(f"First product structure: {type(products[0])}, keys: {products[0].keys() if isinstance(products[0], dict) else 'Not a dict'}")
            
            self.categories = list(set(p.get('category', 'Unknown') for p in products if isinstance(p, dict)))
            prices = [float(p.get('price', 0)) for p in products if isinstance(p, dict) and p.get('price')]
            self.price_range = {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0
            }
        except Exception as e:
            logger.error(f"Error initializing categories and price range: {e}")
            self.categories = ['Unknown']
            self.price_range = {'min': 0, 'max': 0}
        
    def _format_product_for_display(self, product: Dict[str, Any]) -> str:
        """
        Format a product for display in chat responses.
        
        Args:
            product: Product dictionary
            
        Returns:
            Formatted product string
        """
        return f"""**{product['name']}**
- Category: {product['category']}
- Price: ${product['price']}
- Rating: {product['rating']}/5.0
- In Stock: {'Yes' if product['in_stock'] else 'No'}
- Description: {product['description']}
- Features: {product['features']}"""
    
    def search_products(self, query: str, top_k: int = 3) -> str:
        """
        Search for products based on a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Formatted search results
        """
        results = self.query_engine.search(query, top_k=top_k)
        
        if not results:
            return f"I couldn't find any products in our catalog matching '{query}'. Our catalog contains {len(self.products)} products across categories: {', '.join(self.categories)}."
        
        response = f"I found {len(results)} products in our catalog matching '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            product = result["product"]
            similarity = result["similarity"]
            response += f"**#{i}** (Relevance: {similarity:.2f})\n"
            response += self._format_product_for_display(product)
            response += "\n\n"
            
        return response
    
    def filter_products(self, 
                       category: Optional[str] = None,
                       min_price: Optional[float] = None,
                       max_price: Optional[float] = None,
                       in_stock_only: bool = False,
                       min_rating: Optional[float] = None) -> str:
        """
        Filter products based on metadata.
        
        Args:
            category: Filter by product category
            min_price: Minimum price filter
            max_price: Maximum price filter
            in_stock_only: Filter for in-stock products only
            min_rating: Minimum rating filter
            
        Returns:
            Formatted filtered results
        """
        results = self.query_engine.filter_products(
            category=category,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            min_rating=min_rating
        )
        
        if not results:
            filters_applied = []
            if category: filters_applied.append(f"category '{category}'")
            if min_price: filters_applied.append(f"minimum price ${min_price}")
            if max_price: filters_applied.append(f"maximum price ${max_price}")
            if in_stock_only: filters_applied.append("in-stock only")
            if min_rating: filters_applied.append(f"minimum rating {min_rating}")
            
            filter_text = " and ".join(filters_applied) if filters_applied else "your criteria"
            return f"No products in our catalog match {filter_text}. Available categories: {', '.join(self.categories)}. Price range: ${self.price_range['min']:.2f} - ${self.price_range['max']:.2f}."
        
        response = f"Found {len(results)} products in our catalog matching your filters:\n\n"
        
        for i, product in enumerate(results, 1):
            response += f"**#{i}**\n"
            response += self._format_product_for_display(product)
            response += "\n\n"
            
        return response
    
    def search_and_filter(self, 
                         query: str,
                         category: Optional[str] = None,
                         min_price: Optional[float] = None,
                         max_price: Optional[float] = None,
                         in_stock_only: bool = False,
                         min_rating: Optional[float] = None,
                         top_k: int = 3) -> str:
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
            Formatted search and filter results
        """
        results = self.query_engine.search_and_filter(
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            min_rating=min_rating,
            top_k=top_k
        )
        
        if not results:
            return f"No products in our catalog match your search '{query}' with the applied filters. Try broadening your search or adjusting the filters."
        
        response = f"Found {len(results)} products in our catalog matching '{query}' with your filters:\n\n"
        
        for i, result in enumerate(results, 1):
            product = result["product"]
            similarity = result["similarity"]
            response += f"**#{i}** (Relevance: {similarity:.2f})\n"
            response += self._format_product_for_display(product)
            response += "\n\n"
            
        return response
    
    def compare_products(self, product_ids: List[str]) -> str:
        """
        Compare multiple products by their IDs.
        
        Args:
            product_ids: List of product IDs to compare
            
        Returns:
            Comparison of the products
        """
        products_to_compare = []
        found_ids = []
        
        for product_id in product_ids:
            for product in self.products:
                if str(product['product_id']) == product_id:
                    products_to_compare.append(product)
                    found_ids.append(product_id)
                    break
        
        if not products_to_compare:
            available_ids = [str(p['product_id']) for p in self.products[:10]]  # Show first 10 IDs
            return f"No products found with the provided IDs ({', '.join(product_ids)}). Available product IDs include: {', '.join(available_ids)}..."
        
        if len(products_to_compare) < len(product_ids):
            missing_ids = [pid for pid in product_ids if pid not in found_ids]
            response = f"Warning: Could not find products with IDs: {', '.join(missing_ids)}\n\n"
        else:
            response = ""
        
        response += f"Comparing {len(products_to_compare)} products from our catalog:\n\n"
        
        for i, product in enumerate(products_to_compare, 1):
            response += f"**Product {i}:**\n"
            response += self._format_product_for_display(product)
            response += "\n\n"
        
        # Add a simple comparison summary
        if len(products_to_compare) >= 2:
            prices = [float(p['price']) for p in products_to_compare]
            ratings = [float(p['rating']) for p in products_to_compare]
            
            response += "**Quick Comparison:**\n"
            response += f"- Price range: ${min(prices):.2f} - ${max(prices):.2f}\n"
            response += f"- Rating range: {min(ratings):.1f} - {max(ratings):.1f}/5.0\n"
            response += f"- Categories: {', '.join(set(p['category'] for p in products_to_compare))}\n"
        
        return response
    
    def recommend_products(self, user_preferences: str, budget: Optional[float] = None) -> str:
        """
        Recommend products based on user preferences.
        
        Args:
            user_preferences: Description of user preferences
            budget: Maximum budget for recommendations
            
        Returns:
            Product recommendations
        """
        # Filter by budget if provided
        filtered_products = self.products
        if budget is not None:
            filtered_products = [p for p in filtered_products if float(p['price']) <= budget]
        
        if not filtered_products:
            return f"No products in our catalog are within your budget of ${budget}. Our price range is ${self.price_range['min']:.2f} - ${self.price_range['max']:.2f}."
        
        # Use semantic search to find products matching preferences
        results = self.query_engine.search(user_preferences, top_k=min(5, len(filtered_products)))
        
        if not results:
            return f"I couldn't find products matching your preferences '{user_preferences}' in our catalog."
        
        # Filter results by budget if specified
        if budget is not None:
            results = [r for r in results if float(r['product']['price']) <= budget]
        
        if not results:
            return f"No products matching your preferences are within your budget of ${budget}."
        
        budget_text = f" within your budget of ${budget}" if budget else ""
        response = f"Based on your preferences '{user_preferences}'{budget_text}, I recommend these products from our catalog:\n\n"
        
        for i, result in enumerate(results[:3], 1):  # Top 3 recommendations
            product = result["product"]
            similarity = result["similarity"]
            response += f"**Recommendation #{i}** (Match: {similarity:.2f})\n"
            response += self._format_product_for_display(product)
            response += f"\n*Why this matches:* This product scores {similarity:.2f} similarity to your preferences.\n\n"
        
        return response
    
    def chat(self, message: str) -> str:
        """
        Chat with the product search agent.
        This version uses function calling but ensures responses are strictly from the catalog.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": message})
        
        # Prepare system message with strict instructions
        system_message = f"""You are a helpful product search assistant for an e-commerce catalog containing exactly {len(self.products)} products. 
        You can ONLY provide information about products that exist in this catalog. Never invent, hallucinate, or suggest products not in the catalog.
        
        Available categories: {', '.join(self.categories)}
        Price range: ${self.price_range['min']:.2f} - ${self.price_range['max']:.2f}
        
        Use the provided functions to search, filter, compare, and recommend products. Always be helpful but honest about catalog limitations."""
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add chat history (limited to last 10 messages to avoid token limits)
        messages.extend(self.chat_history[-10:])
        
        try:
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=[
                    {
                        "name": "search_products",
                        "description": "Search for products based on a natural language query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of top results to return",
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "filter_products",
                        "description": "Filter products based on metadata",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Filter by product category"
                                },
                                "min_price": {
                                    "type": "number",
                                    "description": "Minimum price filter"
                                },
                                "max_price": {
                                    "type": "number",
                                    "description": "Maximum price filter"
                                },
                                "in_stock_only": {
                                    "type": "boolean",
                                    "description": "Filter for in-stock products only",
                                    "default": False
                                },
                                "min_rating": {
                                    "type": "number",
                                    "description": "Minimum rating filter"
                                }
                            }
                        }
                    },
                    {
                        "name": "search_and_filter",
                        "description": "Search and filter products",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Filter by product category"
                                },
                                "min_price": {
                                    "type": "number",
                                    "description": "Minimum price filter"
                                },
                                "max_price": {
                                    "type": "number",
                                    "description": "Maximum price filter"
                                },
                                "in_stock_only": {
                                    "type": "boolean",
                                    "description": "Filter for in-stock products only",
                                    "default": False
                                },
                                "min_rating": {
                                    "type": "number",
                                    "description": "Minimum rating filter"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of top results to return",
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "compare_products",
                        "description": "Compare multiple products by their IDs",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "product_ids": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of product IDs to compare"
                                }
                            },
                            "required": ["product_ids"]
                        }
                    },
                    {
                        "name": "recommend_products",
                        "description": "Recommend products based on user preferences",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_preferences": {
                                    "type": "string",
                                    "description": "Description of user preferences"
                                },
                                "budget": {
                                    "type": "number",
                                    "description": "Maximum budget for recommendations"
                                }
                            },
                            "required": ["user_preferences"]
                        }
                    }
                ],
                function_call="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to call a function
            if assistant_message.function_call:
                function_name = assistant_message.function_call.name
                function_args = json.loads(assistant_message.function_call.arguments)
                
                # Call the appropriate function
                if function_name == "search_products":
                    function_response = self.search_products(**function_args)
                elif function_name == "filter_products":
                    function_response = self.filter_products(**function_args)
                elif function_name == "search_and_filter":
                    function_response = self.search_and_filter(**function_args)
                elif function_name == "compare_products":
                    function_response = self.compare_products(**function_args)
                elif function_name == "recommend_products":
                    function_response = self.recommend_products(**function_args)
                else:
                    function_response = "Unknown function."
                
                # Return the function response directly (no second GPT call to avoid hallucination)
                final_response = function_response
            else:
                # If no function call, provide a helpful response about catalog capabilities
                final_response = f"""I can help you search our catalog of {len(self.products)} products. Here's what I can do:

- **Search products**: Ask me to find products by name, description, or features
- **Filter products**: Filter by category ({', '.join(self.categories)}), price range (${self.price_range['min']:.2f} - ${self.price_range['max']:.2f}), stock status, or rating
- **Compare products**: Compare specific products by their IDs
- **Get recommendations**: Tell me your preferences and budget for personalized recommendations

What would you like to explore in our catalog?"""
            
            # Add assistant response to chat history
            self.chat_history.append({"role": "assistant", "content": final_response})
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset(self) -> None:
        """
        Reset the agent's chat history.
        """
        self.chat_history = []
