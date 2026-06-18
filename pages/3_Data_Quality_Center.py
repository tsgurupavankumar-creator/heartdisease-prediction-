import streamlit as st
import pandas as pd
from utils.ui_components import page_header, metric_card
from utils.data_loader import load_data, get_data_quality_report

st.set_page_config(page_title="Data Quality Center", layout="wide")

def app():
    page_header("Data Quality Center", "Analyze dataset integrity, missing values, and duplicates")
    
    df = load_data()
    if df is not None:
        report = get_data_quality_report(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Missing Values", str(report.get('missing_values', 0)))
        with col2:
            metric_card("Duplicates", str(report.get('duplicates', 0)))
        with col3:
            metric_card("Total Features", str(report.get('total_columns', 0)))
            
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🔍 Missing Values Analysis")
            if report.get('missing_values', 0) > 0:
                st.warning("Missing values detected.")
                missing_df = pd.DataFrame.from_dict(report.get('missing_cols', {}), orient='index', columns=['Missing Count'])
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found in the dataset.")
                
        with col_right:
            st.markdown("### 👯 Duplicates Analysis")
            if report.get('duplicates', 0) > 0:
                st.warning(f"Found {report.get('duplicates')} duplicate rows.")
                if st.button("View Duplicates"):
                    st.dataframe(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)), use_container_width=True)
            else:
                st.success("No duplicate rows found.")
                
    else:
        st.error("Dataset not available.")

if __name__ == "__main__":
    app()
