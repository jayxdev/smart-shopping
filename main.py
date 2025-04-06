# SmartShopping/main.py
import streamlit as st
import pandas as pd
import sqlite3
from agents.customer_agent import customer_agent
from agents.product_agent import product_agent
from agents.recommendation_agent import recommendation_agent
from utils.data_utils import load_data
import subprocess
import psutil
import time
import signal

def check_ollama_running():
    """Check if Ollama server is already running"""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'ollama':
            return True
    return False

def start_ollama_server():
    """Start Ollama server in background"""
    if not check_ollama_running():
        try:
            # Start Ollama in background
            process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            # Wait briefly for server to start
            time.sleep(3)
            return process
        except Exception as e:
            st.warning(f"Couldn't start Ollama server: {str(e)}")
            return None
    return None

def display_customer_profile(customer_id, users):
    """Display detailed customer profile information"""
    customer = users[users["Customer_ID"] == customer_id].iloc[0]
    
    st.sidebar.subheader("Customer Profile")
    st.sidebar.write(f"**ID:** {customer['Customer_ID']}")
    st.sidebar.write(f"**Age:** {customer['Age']}")
    st.sidebar.write(f"**Gender:** {customer['Gender']}")
    st.sidebar.write(f"**Location:** {customer['Location']}")
    st.sidebar.write(f"**Segment:** {customer['Customer_Segment']}")
    st.sidebar.write(f"**Avg Order Value:** ${customer['Avg_Order_Value']:.2f}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write("**Browsing History**")
        for item in customer["Browsing_History"]:
            st.text(item)
    
    with col2:
        st.write("**Purchase History**")
        for item in customer["Purchase_History"]:
            st.text(item)

def display_recommendations(product_ids, description, confidences, products):
    """Display recommendation results with enhanced product info"""
    st.subheader("🌟 Recommended Products")
    
    # Handle single confidence value or list
    if isinstance(confidences, (float, int)):
        confidences = [confidences] * len(product_ids)
    elif len(confidences) != len(product_ids):
        confidences = [confidences[0]] * len(product_ids)
    
    for i, (pid, confidence) in enumerate(zip(product_ids, confidences), 1):
        product = products[products["Product_ID"] == pid].iloc[0]
        
        with st.expander(f"#{i}: {product['Brand']} | {product['Subcategory']} | Confidence: {confidence*100:.1f}%"):
            # Create two columns (3:1 ratio)
            main_col, metric_col = st.columns([3, 1])
            
            # Main product details
            main_col.write(f"**📦 Category:** {product['Category']}")
            main_col.write(f"**💰 Price:** ${product['Price']:,.2f}")
            main_col.write(f"**⭐ Rating:** {product['Product_Rating']}/5")
            
            # Confidence metrics
            metric_col.metric("Confidence Score", f"{confidence*100:.1f}%")
            
            # Recommendation probability progress bar
            st.progress(
                product["Probability_of_Recommendation"], 
                text=f"Recommendation Probability: {product['Probability_of_Recommendation']*100:.1f}%"
            )
            
            # Sentiment display with color coding
            sentiment = product["Customer_Review_Sentiment_Score"]
            sentiment_label = "Positive" if sentiment > 0.5 else "Negative" if sentiment < -0.5 else "Neutral"
            sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "blue"
            
            st.write(
                f"**😊 Customer Sentiment:** :{sentiment_color}[{sentiment_label} ({sentiment:.2f})]",
            )
            st.caption("Sentiment ranges from -1 (negative) to +1 (positive)")
    
    # Recommendation explanation
    st.subheader("🔍 Recommendation Message")
    if isinstance(description, (list, tuple)):
        st.info(description[0] if description else "No description available")
    else:
        st.info(description)


def run_streamlit(users, products):
    """Main Streamlit application interface"""
    # Try to start Ollama server automatically
    
    ollama_process = start_ollama_server()
    ollama_available = ollama_process is not None
    
    # Set page config
    st.set_page_config(layout="wide", page_title="Smart Shopping Assistant")
    
    k=st.sidebar.slider("Clusters", 2, 24, 12,2, help="Number of clusters for customer segmentation")
    # Add cleanup handler
    def cleanup():
        if ollama_process:
            try:
                ollama_process.terminate()
            except:
                pass
    
    # Register cleanup
    import atexit
    atexit.register(cleanup)

    # Set up Streamlit interface
    st.title("🛍️ Smart Shopping Assistant")
    st.write("AI-powered personalized recommendations based on your shopping behavior")
    
    # Customer selection
    customer_id = st.selectbox(
        "Select Customer", 
        users["Customer_ID"].unique(),
        help="Choose a customer to generate personalized recommendations"
    )
    
    # Display customer profile in sidebar
    display_customer_profile(customer_id, users)
    
    # Recommendation button
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing preferences and finding perfect matches..."):
            try:
                # Get recommendations from agents
                product_ids, description, confidence = recommendation_agent(
                    customer_id, users, products, ollama_available
                )
                
                # Display results
                display_recommendations(product_ids, description, confidence, products)
                
                # Show customer clusters visualization
                _, cluster_img = customer_agent(users,products, n_clusters=k)
                st.subheader("Customer Segmentation Insights")
                st.plotly_chart(cluster_img, use_container_width=True)
                
                # Show recommendation history
                conn = sqlite3.connect("ecommerce.db")
                st.subheader("Your Recommendation History")
                history = pd.read_sql_query(
                    "SELECT Product_ID, Score, Timestamp FROM recommendations "
                    f"WHERE Customer_ID = '{customer_id}' ORDER BY Timestamp DESC LIMIT 5",
                    conn
                )
                
                # Convert timestamp to datetime if needed
                if pd.api.types.is_string_dtype(history["Timestamp"]):
                    history["Timestamp"] = pd.to_datetime(history["Timestamp"], errors="coerce")
                
                if not history.empty:
                    st.dataframe(
                        history.style.format({
                            "Score": "{:.2f}",
                            "Timestamp": lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No recommendation history found")
                
                conn.close()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if "Ollama" in str(e):
                    st.warning("AI descriptions unavailable - using default recommendations")
                    ollama_available = False
                    # Retry without Ollama
                    product_ids, description, confidence = recommendation_agent(
                        customer_id, users, products, False
                    )
                    display_recommendations(product_ids, description, confidence, products)
def main():
    """Main execution function"""
    # Load and preprocess data
    users, products = load_data("data/customer_data_collection.csv", "data/product_recommendation_data.csv")
    
    # Process data through agents
    users, _ = customer_agent(users, products)
    products = product_agent(products)
    
    # Run Streamlit interface
    run_streamlit(users, products)
if __name__ == "__main__":
    main()