import streamlit as st
import json
import os
import plotly.express as px
from utils.ui_components import page_header
from utils.pdf_generator import generate_pdf_report
from config.theme import PRIMARY_COLOR, SECONDARY_COLOR

st.set_page_config(page_title="Model Management", layout="wide")

def app():
    page_header("Model Management", "View active model status and generate clinical reports")
    
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        st.markdown("### Active Model Configuration")
        st.json({
            "Active Model": metadata.get("best_model"),
            "Last Trained": metadata.get("timestamp"),
            "Features Tracked": len(metadata.get("features", [])),
            "Metrics": metadata.get("metrics")
        })
        
        st.markdown("---")
        st.markdown("### 📄 Clinical Report Generation")
        st.info("Generate a comprehensive PDF report of the active model's performance for clinical review.")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Compiling metrics and generating report..."):
                metrics = metadata.get("metrics", {})
                
                # Prepare a temporary image for the report
                try:
                    all_results = metadata.get("all_results", {})
                    best_model = metadata.get("best_model", "")
                    roc_img = None
                    if best_model in all_results:
                        data = all_results[best_model]
                        # Just generating a static report path
                        roc_img = None # In a full system, we'd save the plotly fig to PNG using kaleido
                except Exception as e:
                    pass
                
                try:
                    report_path = generate_pdf_report(metrics, output_path="reports/clinical_report.pdf")
                    st.success(f"Report generated successfully!")
                    
                    with open(report_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                        st.download_button(label="⬇️ Download Clinical Report",
                                        data=PDFbyte,
                                        file_name="Clinical_Heart_Disease_Report.pdf",
                                        mime='application/octet-stream')
                except Exception as e:
                    st.error(f"Failed to generate report. {str(e)}")
                    st.error("Ensure ReportLab is installed correctly.")
                    
    else:
        st.warning("No model has been trained yet. Please visit the Model Training Center.")

if __name__ == "__main__":
    app()
