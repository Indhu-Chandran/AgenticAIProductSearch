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
        
    def _format_product_for_display(self, product: Dict[str, Any]) -> str:
        """
        Format a product for display in chat responses.
        
        Args:
            product: Product dictionary
            
        Returns:
            Formatted product string
        """
        return f"""Product: {product['name']}
        Category: {product['category']}
        Price: ${product['price']}
        Rating: {product['rating']}/5.0
        In Stock: {'Yes' if product['in_stock'] else 'No'}
        Description: {product['description']}
        Features: {product['features']}"""
    
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
            return "No products found matching your query."
        
        response = f"Found {len(results)} products matching your query:\n\n"
        
        for i, result in enumerate(results, 1):
            product = result["product"]
            similarity = result["similarity"]
            response += f"#{i} (Relevance: {similarity:.2f})\n"
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
            return "No products found matching your filters."
        
        response = f"Found {len(results)} products matching your filters:\n\n"
        
        for i, product in enumerate(results, 1):
            response += f"#{i}\n"
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
            return "No products found matching your query and filters."
        
        response = f"Found {len(results)} products matching your query and filters:\n\n"
        
        for i, result in enumerate(results, 1):
            product = result["product"]
            similarity = result["similarity"]
            response += f"#{i} (Relevance: {similarity:.2f})\n"
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
        for product_id in product_ids:
            for product in self.products:
                if str(product['product_id']) == product_id:
                    products_to_compare.append(product)
                    break
        
        if not products_to_compare:
            return "No products found with the provided IDs."
        
        # Use OpenAI to generate a comparison
        product_details = "\n\n".join([self._format_product_for_display(p) for p in products_to_compare])
        
        prompt = f"""Compare the following products and highlight their key differences and similarities:

{product_details}

Provide a detailed comparison focusing on price, features, and overall value."""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful product comparison assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error comparing products: {e}")
            return "Sorry, I encountered an error while comparing these products."
    
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
            return "No products found within your budget."
        
        # Format products for the prompt
        product_details = "\n\n".join([self._format_product_for_display(p) for p in filtered_products])
        
        budget_text = f" with a budget of ${budget}" if budget else ""
        prompt = f"""A customer has the following preferences: {user_preferences}{budget_text}.

Based on these preferences, recommend the most suitable products from the catalog below:

{product_details}

Provide 2-3 recommendations with explanations for why each product matches the customer's needs."""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful product recommendation assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error recommending products: {e}")
            return "Sorry, I encountered an error while generating recommendations."
    
    def chat(self, message: str) -> str:
        """
        Chat with the product search agent.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": message})
        
        # Prepare system message
        system_message = """You are a helpful product search assistant for an e-commerce catalog. 
        Your goal is to help users find products that match their needs and preferences.
        You can search, filter, compare, and recommend products from the catalog.
        Always provide detailed and helpful responses."""
        
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
                
                # Get a new response that incorporates the function result
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        *self.chat_history,
                        assistant_message,
                        {"role": "function", "name": function_name, "content": function_response}
                    ]
                )
                
                final_response = second_response.choices[0].message.content
            else:
                final_response = assistant_message.content
            
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
