import pandas as pd
import ast

def load_data(user_file="data/customer_data_collection.csv", product_file="data/product_recommendation_data.csv"):
    # Load and parse user data
    users = pd.read_csv(user_file)
    users['Browsing_History'] = users['Browsing_History'].apply(ast.literal_eval)
    users['Purchase_History'] = users['Purchase_History'].apply(ast.literal_eval)
    
    # Load and parse product data
    products = pd.read_csv(product_file)
    products['Similar_Product_List'] = products['Similar_Product_List'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    return users, products
