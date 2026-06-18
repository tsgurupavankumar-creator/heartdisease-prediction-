import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import page_header
from config.theme import PLOTLY_TEMPLATE, PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR

st.set_page_config(page_title="Model Comparison", layout="wide")

def app():
    page_header("Model Comparison", "Side-by-side performance comparison of trained models")
    
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        all_results = metadata.get("all_results")
        if not all_results:
            st.warning("Historical comparison data not found. Please retrain models in the Model Training Center.")
            return
            
        # Parse metrics into a dataframe
        metrics_list = []
        for model_name, data in all_results.items():
            row = {"Model": model_name}
            row.update(data["metrics"])
            metrics_list.append(row)
            
        metrics_df = pd.DataFrame(metrics_list)
        
        st.markdown("### Performance Metrics Comparison")
        st.dataframe(metrics_df.set_index("Model").style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Recall Comparison")
            fig_recall = px.bar(
                metrics_df.sort_values("Recall", ascending=False), 
                x="Model", y="Recall", 
                color="Model", 
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
            )
            fig_recall.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_recall, use_container_width=True)
            
        with col2:
            st.markdown("### ROC-AUC Comparison")
            fig_auc = px.bar(
                metrics_df.sort_values("ROC-AUC", ascending=False), 
                x="Model", y="ROC-AUC", 
                color="Model", 
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
            )
            fig_auc.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_auc, use_container_width=True)
            
        st.markdown("### Multi-Model ROC Curves")
        fig_roc = go.Figure()
        colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
        
        for idx, (model_name, data) in enumerate(all_results.items()):
            roc_data = data.get("roc_data", {})
            fpr = roc_data.get("fpr", [])
            tpr = roc_data.get("tpr", [])
            auc = data["metrics"].get("ROC-AUC", 0)
            
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                         name=f"{model_name} (AUC = {auc:.3f})",
                                         line=dict(color=colors[idx % len(colors)], width=2)))
                                         
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
            
    else:
        st.warning("No model has been trained yet. Please visit the Model Training Center.")

if __name__ == "__main__":
    app()
