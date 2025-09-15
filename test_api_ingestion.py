#!/usr/bin/env python3
"""
Test script to verify the updated ingestion works with the new API format.
"""

import os
from dotenv import load_dotenv
from ingest import ProductCatalogIngester

# Load environment variables
load_dotenv()

def test_api_ingestion():
    """Test ingestion from the new API endpoint"""
    print("=== Testing API Ingestion ===")
    
    api_url = "https://productdataservice.vercel.app/api/products"
    
    # Initialize ingester
    ingester = ProductCatalogIngester(persist_dir="./storage_api_test")
    
    try:
        print(f"Fetching products from: {api_url}")
        products = ingester.load_from_url(api_url)
        
        print(f"✅ Successfully loaded {len(products)} products")
        
        # Show first few products
        print("\nFirst 3 products:")
        for i, product in enumerate(products[:3]):
            print(f"{i+1}. {product['name']} (ID: {product['product_id']})")
            print(f"   Category: {product['category']}")
            print(f"   Price: ${product['price']}")
            print(f"   In Stock: {product['in_stock']}")
            print(f"   Description: {product['description'][:100]}...")
            print()
        
        # Show categories
        categories = list(set(p['category'] for p in products))
        print(f"Categories found: {', '.join(categories)}")
        
        # Test full ingestion and indexing
        print("\n=== Testing Full Ingestion with Embeddings ===")
        success = ingester.ingest_and_index(source_url=api_url)
        
        if success:
            print("✅ Successfully ingested and created embeddings")
            
            # Load back the data
            loaded_products, loaded_embeddings = ingester.load_data()
            if loaded_products and loaded_embeddings:
                print(f"✅ Verified: {len(loaded_products)} products and {len(loaded_embeddings)} embeddings saved")
            else:
                print("❌ Failed to load back the saved data")
        else:
            print("❌ Failed to ingest and create embeddings")
            
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_ingestion()
