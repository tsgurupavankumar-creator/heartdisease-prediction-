import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from utils.ui_components import page_header
from config.theme import PLOTLY_TEMPLATE, PRIMARY_COLOR, SECONDARY_COLOR

st.set_page_config(page_title="Feature Importance", layout="wide")

def app():
    page_header("Feature Importance", "Interpret model decision drivers to understand clinical risks")
    
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        best_model = metadata.get("best_model", "")
        features = metadata.get("features", [])
        all_results = metadata.get("all_results", {})
        
        if best_model and best_model in all_results:
            importance = all_results[best_model].get("feature_importance")
            
            if importance and len(importance) == len(features):
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values("Importance", ascending=True)
                
                st.markdown(f"### Feature Impact - **{best_model}**")
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale=[SECONDARY_COLOR, PRIMARY_COLOR]
                )
                fig.update_layout(template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("Features with higher importance scores had a greater influence on the model's predictions.")
                
                st.markdown("---")
                st.markdown("### SHAP Explainability (Global Summary)")
                st.info("SHAP values provide medical interpretability by showing the directional impact of each feature across the dataset.")
                
                import joblib
                import matplotlib.pyplot as plt
                
                try:
                    if not os.path.exists("models/X_train_scaled.joblib"):
                        st.warning("SHAP background data is missing. Please navigate to the Model Training Center and retrain the model.")
                    else:
                        X_train_scaled = joblib.load("models/X_train_scaled.joblib")
                        model_obj = joblib.load(f"models/{best_model.replace(' ', '_')}.joblib")
                        
                        from utils.visualization import plot_shap_summary
                        fig_shap, _, _ = plot_shap_summary(model_obj, X_train_scaled, features)
                        
                        if fig_shap:
                            st.pyplot(fig_shap)
                        else:
                            st.warning("SHAP explanation not currently supported for this calibrated model ensemble.")
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {str(e)}")
                    
            else:
                st.warning(f"Feature importance is not supported or not available for {best_model}.")
                
    else:
        st.warning("No model has been trained yet. Please visit the Model Training Center.")

if __name__ == "__main__":
    app()
