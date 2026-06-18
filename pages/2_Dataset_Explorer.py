import streamlit as st
from utils.ui_components import page_header
from utils.data_loader import load_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

def app():
    page_header("Dataset Explorer", "Inspect the raw dataset and feature schemas")
    
    df = load_data()
    if df is not None:
        st.markdown("### Raw Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Feature Data Types")
            dtypes_df = df.dtypes.astype(str).reset_index()
            dtypes_df.columns = ["Feature", "Data Type"]
            st.dataframe(dtypes_df, use_container_width=True)
            
        with col2:
            st.markdown("### Descriptive Statistics")
            st.dataframe(df.describe().T, use_container_width=True)
            
        st.info(f"Showing sample of dataset. Total records available: {len(df)}")
    else:
        st.error("Please ensure heart.csv is available in the root directory.")

if __name__ == "__main__":
    app()
