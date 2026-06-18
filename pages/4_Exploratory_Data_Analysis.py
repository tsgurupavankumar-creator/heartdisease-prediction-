import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import page_header
from utils.data_loader import load_data, preprocess_data, detect_target_column
from utils.visualization import plot_target_distribution, plot_correlation_heatmap

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

def app():
    page_header("Exploratory Data Analysis", "Visualize dataset distributions and feature correlations")
    
    df = load_data()
    if df is not None:
        clean_df = preprocess_data(df)
        target_col = detect_target_column(clean_df)
        
        st.markdown("### Target Variable Distribution")
        if target_col:
            fig_target = plot_target_distribution(clean_df, target_col)
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.warning("Could not detect target column for distribution plot.")
            
        st.markdown("### Feature Correlation Matrix")
        fig_corr = plot_correlation_heatmap(clean_df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("### Feature Distributions")
        num_cols = clean_df.select_dtypes(include=['number']).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
            
        selected_feature = st.selectbox("Select a feature to visualize its distribution:", num_cols)
        
        if selected_feature:
            if target_col:
                fig_dist = px.histogram(clean_df, x=selected_feature, color=target_col, marginal="box", 
                                        color_discrete_sequence=["#1E88E5", "#E53935"],
                                        title=f"Distribution of {selected_feature} by {target_col}")
            else:
                fig_dist = px.histogram(clean_df, x=selected_feature, marginal="box", 
                                        color_discrete_sequence=["#1E88E5"],
                                        title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig_dist, use_container_width=True)
            
    else:
        st.error("Dataset not available.")

if __name__ == "__main__":
    app()
