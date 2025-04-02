# SmartShopping/agents/recommendation_agent.py
import pandas as pd
import numpy as np
import sqlite3
import json
import ollama
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def recommendation_agent(customer_id, users, products, ollama_enabled=True):
    conn = sqlite3.connect("ecommerce.db")
    
    try:
        # Get user cluster and data
        user_cluster = pd.read_sql_query(
            f"SELECT cluster FROM user_clusters WHERE Customer_ID = '{customer_id}'", 
            conn
        ).iloc[0]["cluster"]
        
        user_data = users[users["Customer_ID"] == customer_id].iloc[0]
        purchase_history = user_data["Purchase_History"]
        browsing_history = user_data["Browsing_History"]

        # Load product similarity data
        similar_products = pd.read_sql_query(
            "SELECT Product_ID, Similar_Product_List FROM product_similarity", 
            conn
        )
        similar_products["Similar_Product_List"] = similar_products["Similar_Product_List"].apply(json.loads)
        products = products.merge(similar_products, on="Product_ID", how="left")

        # Dynamic category preference calculation
        def calculate_category_weights(history):
            if not history:
                return {}
            category_counts = pd.Series(history).value_counts(normalize=True)
            return category_counts.to_dict()

        purchase_weights = calculate_category_weights(purchase_history)
        browse_weights = calculate_category_weights(browsing_history)
        
        # Combined category preference (60% purchase, 40% browsing)
        all_categories = set(purchase_weights.keys()).union(set(browse_weights.keys()))
        category_preference = {
            cat: (0.6 * purchase_weights.get(cat, 0) + 0.4 * browse_weights.get(cat, 0))
            for cat in all_categories
        }

        # Calculate base recommendation scores with proper normalization
        products["recommendation_score"] = (
            (0.35 * (products["Product_Rating"] / 5)) +  # Normalized to 0-0.35
            (0.25 * products["Probability_of_Recommendation"]) +  # Already 0-1
            (0.2 * ((products["Customer_Review_Sentiment_Score"] + 1) / 2)) +  # -1 to 1 -> 0 to 1
            (0.1 * (products["Average_Rating_of_Similar_Products"] / 5))  # Normalized to 0-0.1
        )

        # Apply dynamic category boosting (0-0.1 additional)
        for cat, weight in category_preference.items():
            products.loc[products["Category"] == cat, "recommendation_score"] += (0.1 * weight)

        # Add purchase history similarity boost (0-0.2 additional)
        if purchase_history:
            tfidf = TfidfVectorizer()
            product_texts = (
                products["Category"] + " " + 
                products["Subcategory"] + " " + 
                products["Brand"]
            )
            tfidf_matrix = tfidf.fit_transform(product_texts)
            purchase_text = " ".join(purchase_history)
            purchase_vec = tfidf.transform([purchase_text])
            similarities = cosine_similarity(purchase_vec, tfidf_matrix).flatten()
            products["recommendation_score"] += (0.2 * similarities)

        # Ensure scores are between 0-1 (0-100%)
        products["recommendation_score"] = products["recommendation_score"].clip(0, 1)
        
        # Get top 3 recommendations with individual confidence scores
        recommended = products.nlargest(3, "recommendation_score")
        confidences = recommended["recommendation_score"].tolist()
        
        # Generate description for top product
        top_product = recommended.iloc[0]
        preferred_category = max(category_preference.items(), key=lambda x: x[1])[0] if category_preference else "General"
        
        if ollama_enabled:
            try:
                prompt = (
                    f"Create a 2-3 sentence sugercoated informal recommendation for a customer who likes {preferred_category}. "
                    f"Product: {top_product['Brand']} {top_product['Subcategory']} "
                    f"(Rating: {top_product['Product_Rating']}/5, Price: ${top_product['Price']}). "
                    f"Customer sentiment: {top_product['Customer_Review_Sentiment_Score']:.1f}."
                )
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                ai_description = response["message"]["content"]
            except Exception:
                ai_description = generate_fallback_description(top_product)
        else:
            ai_description = generate_fallback_description(top_product)

        # Store recommendations
        recommendations = pd.DataFrame({
            "Customer_ID": [customer_id] * len(recommended),
            "Product_ID": recommended["Product_ID"].tolist(),
            "Score": confidences,
            "Timestamp": [datetime.now()] * len(recommended)
        })
        recommendations.to_sql("recommendations", conn, if_exists="append", index=False)
        
        return (
            recommended["Product_ID"].tolist(),
            ai_description,
            confidences  # Return list of individual confidence scores
        )
        
    finally:
        conn.close()

def generate_fallback_description(product):
    return (
        f"Recommended: {product['Brand']} {product['Subcategory']}\n"
        f"• Rating: {product['Product_Rating']}/5\n"
        f"• Price: ${product['Price']}\n"
        f"• Customer Sentiment: {product['Customer_Review_Sentiment_Score']:.1f}"
    )