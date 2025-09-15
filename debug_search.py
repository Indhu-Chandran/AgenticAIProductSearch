#!/usr/bin/env python3
"""
Debug script to test product search and identify issues with catalog results.
"""

import os
import json
from dotenv import load_dotenv
from ingest import ProductCatalogIngester
from agent import ProductSearchAgent
from query_engine import ProductQueryEngine

# Load environment variables
load_dotenv()

def test_catalog_loading():
    """Test if catalog is properly loaded"""
    print("=== Testing Catalog Loading ===")
    
    # Load from storage
    ingester = ProductCatalogIngester(persist_dir="./storage")
    products, embeddings = ingester.load_data()
    
    if products and embeddings:
        print(f"✓ Loaded {len(products)} products and {len(embeddings)} embeddings")
        
        # Show first few products
        print("\nFirst 3 products in catalog:")
        for i, product in enumerate(products[:3]):
            print(f"{i+1}. {product['name']} (ID: {product['product_id']}) - ${product['price']}")
        
        return products, embeddings
    else:
        print("✗ Failed to load products and embeddings")
        return None, None

def test_direct_search(products, embeddings):
    """Test direct search without agent"""
    print("\n=== Testing Direct Search ===")
    
    query_engine = ProductQueryEngine(products, embeddings)
    
    # Test search
    test_query = "smartphone"
    results = query_engine.search(test_query, top_k=3)
    
    print(f"Search query: '{test_query}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results):
        product = result["product"]
        similarity = result["similarity"]
        print(f"{i+1}. {product['name']} (ID: {product['product_id']}) - Similarity: {similarity:.3f}")
    
    return results

def test_agent_search(products, embeddings):
    """Test agent search"""
    print("\n=== Testing Agent Search ===")
    
    agent = ProductSearchAgent(products, embeddings)
    
    # Test direct method call (bypassing chat)
    test_query = "smartphone"
    response = agent.search_products(test_query, top_k=3)
    
    print(f"Agent search response for '{test_query}':")
    print(response)
    
    return response

def test_agent_chat(products, embeddings):
    """Test agent chat interface"""
    print("\n=== Testing Agent Chat ===")
    
    agent = ProductSearchAgent(products, embeddings)
    
    # Test chat
    test_message = "Show me smartphones"
    response = agent.chat(test_message)
    
    print(f"Chat message: '{test_message}'")
    print(f"Chat response:")
    print(response)
    
    return response

def main():
    print("Product Search Debug Tool")
    print("=" * 50)
    
    # Test 1: Load catalog
    products, embeddings = test_catalog_loading()
    
    if not products or not embeddings:
        print("Cannot proceed without loaded data. Please ingest your catalog first.")
        return
    
    # Test 2: Direct search
    direct_results = test_direct_search(products, embeddings)
    
    # Test 3: Agent search (direct method)
    agent_response = test_agent_search(products, embeddings)
    
    # Test 4: Agent chat
    chat_response = test_agent_chat(products, embeddings)
    
    print("\n=== Summary ===")
    print("If you see products not in your catalog in the chat response,")
    print("the issue is likely in the agent's chat method using GPT to generate")
    print("responses that go beyond your actual catalog data.")

if __name__ == "__main__":
    main()
