import streamlit as st
import requests
from utils.logger import Logger

logger = Logger('streamlit_app', level="DEBUG").get_logger()

st.set_page_config(page_title="Olist Churn Predictor", layout="centered")

st.title("📊 Olist Customer Churn Prediction")

with st.form("churn_form"):
    st.subheader("Customer Input Features")
    customer_id = st.text_input("Customer Unique ID", value="cust_123")
    
    col1, col2 = st.columns(2)
    with col1:
        # Step=1 se ye integer inputs ban jayenge
        frequency = st.number_input("Frequency", min_value=1, value=1, step=1)
        monetary = st.number_input("Monetary Value", min_value=0.0, value=100.0)
        review_score = st.slider("Review Score", 1, 5, 4)
        
    with col2:
        installments = st.number_input("Installments", min_value=1, value=1, step=1)
        total_payment = st.number_input("Total Payment", min_value=0.0, value=150.0)
        submit_button = st.form_submit_button("Predict Churn Status")

if submit_button:
    # Explicitly int() mein convert kar rahe hain payload ke liye
    payload = {
        "customer_id": customer_id,
        "frequency": int(frequency),
        "monetary": float(monetary),
        "review_score": int(review_score),
        "installments": int(installments),
        "total_payment": float(total_payment)
    }
    
    try:
        response = requests.post(
                                "https://soraubh7march-churn-api.hf.space/predict",
                                json=payload,
                                timeout=30
)
        
        if response.status_code == 200:
            try:
                result = response.json()
        
                if result.get("status") != "success":
                    st.error("Prediction failed on API side")
                else:
                    st.divider()
        
                    if result['prediction'] == 1:
                        st.error("⚠️ High Risk: Customer might Churn!")
                    else:
                        st.success("✅ Low Risk: Customer is likely to Stay.")
        
                    st.metric(
                        "Churn Probability",
                        f"{round(result['probability'] * 100, 2)}%"
                    )
        
            except Exception:
                st.error("⚠️ API is waking up... try again in 10–20 seconds.")
        
        else:
            st.error(f"API Error: {response.text}")
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        st.error("Check if FastAPI is running.")



        