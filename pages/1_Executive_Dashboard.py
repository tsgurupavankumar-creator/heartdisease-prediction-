import streamlit as st
import os
import json
from utils.ui_components import page_header, metric_card
from utils.data_loader import load_data, get_data_quality_report, detect_target_column

st.set_page_config(page_title="Executive Dashboard", layout="wide")

def app():
    page_header("Executive Dashboard", "High-level overview of dataset readiness and model status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Status")
        df = load_data()
        if df is not None:
            target_col = detect_target_column(df)
            report = get_data_quality_report(df)
            
            c1, c2 = st.columns(2)
            with c1:
                metric_card("Total Records", f"{report.get('total_rows', 0):,}")
                metric_card("Missing Values", str(report.get('missing_values', 0)))
            with c2:
                metric_card("Total Features", str(report.get('total_columns', 0)))
                metric_card("Duplicate Rows", str(report.get('duplicates', 0)))
                
            if target_col:
                st.success(f"Target column automatically detected: **{target_col}**")
            else:
                st.warning("Target column could not be automatically detected.")
        else:
            st.error("Dataset not loaded.")
            
    with col2:
        st.markdown("### 🤖 Model Status")
        if os.path.exists("models/model_metadata.json"):
            with open("models/model_metadata.json", "r") as f:
                metadata = json.load(f)
                
            st.success(f"Active Model: **{metadata.get('best_model', 'Unknown')}**")
            metrics = metadata.get('metrics', {})
            
            c1, c2 = st.columns(2)
            with c1:
                metric_card("ROC-AUC Score", f"{metrics.get('ROC-AUC', 0):.4f}")
                metric_card("Recall (Sensitivity)", f"{metrics.get('Recall', 0):.4f}")
            with c2:
                metric_card("F1 Score", f"{metrics.get('F1 Score', 0):.4f}")
                metric_card("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
                
            st.info(f"Last trained: {metadata.get('timestamp', 'Unknown')}")
        else:
            st.warning("No trained model found. Please proceed to the Model Training Center.")

if __name__ == "__main__":
    app()
