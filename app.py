import streamlit as st
import os

from utils.ui_components import apply_enterprise_theme

# Must be the first Streamlit command
st.set_page_config(
    page_title="Heart Disease Prediction Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global theme
apply_enterprise_theme()

def main():
    st.sidebar.title("🫀 Heart Analytics Platform")
    st.sidebar.markdown("---")
    
    st.title("Executive Overview")
    st.markdown("""
    Welcome to the **Healthcare Analytics SaaS Platform** for Heart Disease Risk Prediction.
    
    This enterprise-grade application provides comprehensive tools for analyzing cardiovascular dataset metrics, 
    training robust machine learning models, evaluating clinical reliability, and running patient risk predictions.
    
    ### System Modules:
    
    * **1. Executive Dashboard**: High-level overview of dataset and model readiness.
    * **2. Dataset Explorer**: Inspect raw dataset and schema.
    * **3. Data Quality Center**: Check for duplicates, missing values, and data integrity.
    * **4. Exploratory Data Analysis**: Visualize distributions, correlations, and target relationships.
    * **5. Model Training Center**: Configure hyper-parameters and train ML algorithms.
    * **6. Model Evaluation Center**: Detailed classification metrics (ROC-AUC, Recall).
    * **7. Model Comparison**: Side-by-side performance comparison.
    * **8. Feature Importance**: Interpret model decision drivers.
    * **9. Patient Risk Prediction**: Interactive interface for real-time inference.
    * **10. Model Management**: View metadata and operational status.
    * **11. Settings**: System configuration and logs.
    
    Please navigate using the sidebar to begin your clinical analysis workflow.
    """)
    
    st.info("System Ready. Dataset `heart.csv` is configured as the primary data source.", icon="✅")

if __name__ == "__main__":
    main()
