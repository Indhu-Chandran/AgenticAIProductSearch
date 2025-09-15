#!/usr/bin/env python3
"""
Local test script to verify the fixed agent only returns products from your catalog.
"""

import os
import json
from dotenv import load_dotenv
from ingest import ProductCatalogIngester
from agent_fixed import ProductSearchAgent

# Load environment variables
load_dotenv()

def test_local_agent():
    """Test the fixed agent with local catalog data"""
    print("=== Testing Fixed Agent Locally ===")
    
    # Load from storage
    ingester = ProductCatalogIngester(persist_dir="./storage")
    products, embeddings = ingester.load_data()
    
    if not products or not embeddings:
        print("❌ No data found. Please ingest your catalog first.")
        return
    
    print(f"✅ Loaded {len(products)} products and {len(embeddings)} embeddings")
    
    # Initialize the fixed agent
    agent = ProductSearchAgent(products, embeddings)
    
    # Test queries that might cause hallucination
    test_queries = [
        "Show me laptops",  # Not in your catalog
        "I need a gaming chair",  # Not in your catalog
        "Find me smartphones",  # In your catalog
        "Show me TVs",  # In your catalog
        "I want a coffee maker",  # Not exactly in catalog
        "Kitchen appliances under $200",  # Filter test
        "Electronics with high ratings"  # Category + filter test
    ]
    
    print("\n" + "="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            response = agent.chat(query)
            print(response)
            
            # Check if response mentions products not in catalog
            if "laptop" in response.lower() and not any("laptop" in p['name'].lower() for p in products):
                print("⚠️  WARNING: Response mentions laptops but none in catalog!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print(f"\n✅ Test completed. The agent should only mention products from your {len(products)}-item catalog.")

if __name__ == "__main__":
    test_local_agent()
