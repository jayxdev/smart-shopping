# SmartShopping/agents/product_agent.py
import sqlite3
import pandas as pd
import json

def product_agent(products):
    conn = sqlite3.connect("ecommerce.db")
    
    # Create a copy of products to avoid modifying the original
    products_db = products.copy()
    
    # Convert Similar_Product_List to JSON string
    products_db["Similar_Product_List"] = products_db["Similar_Product_List"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else '[]'
    )
    
    # Store main product information
    products_db[[
        "Product_ID", "Category", "Subcategory", "Price", "Brand",
        "Product_Rating", "Customer_Review_Sentiment_Score",
        "Probability_of_Recommendation"
    ]].to_sql("products", conn, if_exists="replace", index=False)
    
    # Store similarity information
    products_db[["Product_ID", "Similar_Product_List"]].to_sql(
        "product_similarity", conn, if_exists="replace", index=False)
    
    # Create tables with proper schema if they don't exist
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            Product_ID TEXT PRIMARY KEY,
            Category TEXT,
            Subcategory TEXT,
            Price REAL,
            Brand TEXT,
            Product_Rating REAL,
            Customer_Review_Sentiment_Score REAL,
            Probability_of_Recommendation REAL
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS product_similarity (
            Product_ID TEXT PRIMARY KEY,
            Similar_Product_List TEXT,
            FOREIGN KEY (Product_ID) REFERENCES products (Product_ID)
        )
        """)
        # the recommendations table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            Customer_ID TEXT,
            Product_ID TEXT,
            Score REAL,
            Timestamp DATETIME,
            FOREIGN KEY (Customer_ID) REFERENCES user_clusters (Customer_ID),
            FOREIGN KEY (Product_ID) REFERENCES products (Product_ID)
        )
        """)
    
    conn.close()
    return products