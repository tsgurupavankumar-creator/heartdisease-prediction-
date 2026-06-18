"""
Model Evaluation Center
Detailed performance metrics of the currently active model
Dark Mode Adaptive - Uses only native Streamlit components
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from utils.ui_components import page_header, metric_card
from utils.ml_pipeline import load_model

# Page configuration
st.set_page_config(
    page_title="Model Evaluation Center",
    page_icon="📊",
    layout="wide"
)


def plot_calibration_curve(prob_true, prob_pred, model_name):
    """Plot calibration curve using Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='lines+markers',
        name=f'{model_name} (Calibrated)',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=10, color='#1E88E5')
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='#E53935', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        height=400,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_precision_recall_curve(precision, recall, model_name):
    """Plot Precision-Recall curve using Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'{model_name} (PR Curve)',
        line=dict(color='#1E88E5', width=3)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Plot ROC curve using Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.4f})',
        line=dict(color='#1E88E5', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.500)',
        line=dict(color='#E53935', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        height=400,
        template='plotly_white',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def app():
    """Main application function."""
    
    page_header(
        "Model Evaluation Center",
        "Detailed performance metrics of the currently active model"
    )
    
    # Check for trained model
    metadata_path = "models/model_metadata.json"
    
    if not os.path.exists(metadata_path):
        st.warning(
            "⚠️ No model has been trained yet. "
            "Please visit the **Model Training Center** to train a model first."
        )
        return
    
    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load metadata: {str(e)}")
        return
    
    # Extract ALL metadata fields
    best_model = metadata.get("best_model", "Unknown")
    model_type = metadata.get("model_type", "Unknown")
    training_samples = metadata.get("training_samples", 0)
    metrics = metadata.get("metrics", {})
    timestamp = metadata.get("timestamp", "Unknown")
    features = metadata.get("features", [])
    all_results = metadata.get("all_results", {})
    
    # Display model information using native Streamlit components
    st.markdown("### 🧠 Active Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", best_model)
    with col2:
        st.metric("Type", model_type)
    with col3:
        st.metric("Training Samples", training_samples)
    with col4:
        st.metric("Trained", timestamp[:16] if timestamp != "Unknown" else "Unknown")
    
    st.divider()
    
    # ============================================
    # CORE CLINICAL METRICS
    # ============================================
    st.markdown("### 🏥 Core Clinical Metrics")
    
    st.info(
        "💡 **Healthcare Priority:** In medical contexts, maximizing **Recall** "
        "(Sensitivity) ensures fewer false negatives (missing sick patients). "
        "**ROC-AUC** balances overall discrimination across thresholds."
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Recall (Sensitivity)",
            f"{metrics.get('Recall', 0):.4f}"
        )
    
    with col2:
        st.metric(
            "📈 ROC-AUC",
            f"{metrics.get('ROC-AUC', 0):.4f}"
        )
    
    with col3:
        st.metric(
            "⚖️ F1 Score",
            f"{metrics.get('F1 Score', 0):.4f}"
        )
    
    with col4:
        st.metric(
            "🎯 Precision",
            f"{metrics.get('Precision', 0):.4f}"
        )
    
    with col5:
        st.metric(
            "📊 Brier Score",
            f"{metrics.get('Brier_Score', 0):.4f}"
        )
    
    # Cross-validation metrics
    cv_mean = metrics.get('CV_ROC_AUC_Mean', None)
    cv_std = metrics.get('CV_ROC_AUC_STD', None)
    
    if cv_mean is not None:
        st.info(f"🔬 **Cross-Validation (5-Fold):** Mean ROC-AUC = {cv_mean:.4f} ± {cv_std:.4f}")
    
    st.divider()
    
    # ============================================
    # ADVANCED EVALUATION CURVES
    # ============================================
    st.markdown("### 📊 Advanced Evaluation Curves")
    
    if best_model in all_results:
        model_data = all_results[best_model]
        
        # Create tabs for different curves
        tab1, tab2, tab3 = st.tabs(["ROC Curve", "Calibration Curve", "Precision-Recall Curve"])
        
        with tab1:
            # ROC Curve
            roc_data = model_data.get("roc_data", {})
            if roc_data and roc_data.get("fpr") and roc_data.get("tpr"):
                fig = plot_roc_curve(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    metrics.get("ROC-AUC", 0),
                    best_model
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation using native components
                roc_auc = metrics.get("ROC-AUC", 0)
                if roc_auc >= 0.90:
                    st.success("✅ **Excellent discrimination** - Model has outstanding ability to distinguish between classes.")
                elif roc_auc >= 0.80:
                    st.info("ℹ️ **Good discrimination** - Model has good ability to distinguish between classes.")
                elif roc_auc >= 0.70:
                    st.warning("⚠️ **Fair discrimination** - Model has moderate ability to distinguish between classes.")
                else:
                    st.error("❌ **Poor discrimination** - Model may need improvement.")
            else:
                st.info("ROC curve data not available.")
        
        with tab2:
            # Calibration Curve
            calib_data = model_data.get("calib_data", {})
            if calib_data and calib_data.get("prob_true") and calib_data.get("prob_pred"):
                fig = plot_calibration_curve(
                    calib_data["prob_true"],
                    calib_data["prob_pred"],
                    best_model
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calibration interpretation using native components
                brier = metrics.get("Brier_Score", 1.0)
                if brier < 0.10:
                    st.success("✅ **Excellent calibration** - Predicted probabilities are highly reliable.")
                elif brier < 0.20:
                    st.info("ℹ️ **Good calibration** - Predicted probabilities are reasonably reliable.")
                elif brier < 0.30:
                    st.warning("⚠️ **Moderate calibration** - Predicted probabilities have some uncertainty.")
                else:
                    st.error("❌ **Poor calibration** - Predicted probabilities may be unreliable.")
            else:
                st.info("Calibration curve data not available.")
        
        with tab3:
            # Precision-Recall Curve
            pr_data = model_data.get("pr_data", {})
            if pr_data and pr_data.get("precision") and pr_data.get("recall"):
                fig = plot_precision_recall_curve(
                    pr_data["precision"],
                    pr_data["recall"],
                    best_model
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # PR AUC interpretation using native components
                precision = metrics.get("Precision", 0)
                recall = metrics.get("Recall", 0)
                if precision > 0.80 and recall > 0.80:
                    st.success("✅ **Excellent balance** - Model maintains high precision and recall.")
                elif precision > 0.70 and recall > 0.70:
                    st.info("ℹ️ **Good balance** - Model maintains reasonable precision and recall.")
                else:
                    st.warning("⚠️ **Trade-off detected** - Consider adjusting threshold for better balance.")
            else:
                st.info("Precision-Recall curve data not available.")
    
    st.divider()
    
    # ============================================
    # CONFUSION MATRIX
    # ============================================
    st.markdown("### 📋 Confusion Matrix")
    
    if best_model in all_results:
        model_data = all_results[best_model]
        cm = model_data.get("confusion_matrix", [])
        
        if cm and len(cm) == 2:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual No Disease', 'Actual Disease'],
                    columns=['Predicted No Disease', 'Predicted Disease']
                )
                st.dataframe(cm_df, use_container_width=True)
            
            with col2:
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                total = tn + fp + fn + tp
                
                # Calculate metrics
                accuracy = (tn + tp) / total if total > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                st.markdown("**Matrix Interpretation:**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("True Negatives", tn)
                    st.metric("False Positives", fp, delta="Type I Error", delta_color="inverse")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Specificity", f"{specificity:.2%}")
                with col_b:
                    st.metric("True Positives", tp)
                    st.metric("False Negatives", fn, delta="Type II Error", delta_color="inverse")
                    st.metric("Sensitivity", f"{sensitivity:.2%}")
                    st.metric("PPV", f"{ppv:.2%}")
    
    st.divider()
    
    # ============================================
    # ERROR ANALYSIS
    # ============================================
    st.markdown("### 🔍 Error Analysis")
    
    st.info(
        "💡 **Clinical Focus:** Understanding misclassifications helps improve "
        "the model's clinical utility. **False Negatives** are particularly important "
        "as they represent missed diagnoses."
    )
    
    if best_model in all_results:
        model_data = all_results[best_model]
        
        fp_count = len(model_data.get("false_positives", []))
        fn_count = len(model_data.get("false_negatives", []))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"""
            **⚠️ False Negatives (Type II Error)**
            
            **{fn_count}** patients
            
            Patients predicted as **low risk** but actually had **heart disease**.
            
            ❌ These are the most clinically dangerous errors.
            """)
        
        with col2:
            st.warning(f"""
            **⚠️ False Positives (Type I Error)**
            
            **{fp_count}** patients
            
            Patients predicted as **high risk** but actually had **no disease**.
            
            ⚠️ These cause unnecessary patient anxiety and testing.
            """)
        
        # Error ratio
        total_errors = fp_count + fn_count
        total_predictions = fp_count + fn_count + metrics.get('Accuracy', 0) * 100
        
        if total_predictions > 0:
            error_rate = total_errors / total_predictions
            st.info(f"📊 **Error Statistics:** Total Errors: {total_errors} | Error Rate: {error_rate:.2%}")
        
        # Recommendations - using native components
        st.markdown("#### 💡 Recommendations for Model Improvement")
        st.markdown("""
        - **High False Negatives:** Consider lowering the decision threshold to increase sensitivity.
        - **High False Positives:** Consider increasing the threshold or adding more discriminative features.
        - Investigate patient subgroups (age, cholesterol ranges, etc.) where the model fails most often.
        - Consider ensemble methods or additional features for hard-to-classify cases.
        """)
    else:
        st.info("Error analysis data not available.")
    
    st.divider()
    
    # ============================================
    # FEATURE IMPORTANCE
    # ============================================
    st.markdown("### 🔬 Feature Importance")
    
    if best_model in all_results:
        model_data = all_results[best_model]
        importance = model_data.get("feature_importance")
        
        if importance and features:
            importance_df = pd.DataFrame({
                "Feature": features[:len(importance)],
                "Importance": importance
            }).sort_values("Importance", ascending=False)
            
            # Display chart
            st.bar_chart(importance_df.set_index("Feature"))
            
            # Top 5 features
            st.markdown("#### Top 5 Risk Factors")
            for _, row in importance_df.head(5).iterrows():
                st.write(f"**{row['Feature']}**: {row['Importance']:.3f}")
            
            # Clinical interpretation
            st.info(
                "💡 **Clinical Interpretation:** Features with higher importance "
                "have stronger influence on the prediction. Consider these factors "
                "when evaluating patient risk."
            )
        else:
            st.info("Feature importance not available for this model.")
    
    st.divider()
    
    # ============================================
    # MODEL METADATA
    # ============================================
    st.markdown("### 📋 Model Metadata")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model)
    
    with col2:
        st.metric("Model Type", model_type)
    
    with col3:
        st.metric("Training Samples", training_samples)
    
    with col4:
        st.metric("Features", len(features))
    
    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ **Medical Disclaimer:** This tool is intended for educational and "
        "analytical purposes only. It is not a substitute for professional "
        "medical diagnosis, treatment, or healthcare advice."
    )


if __name__ == "__main__":
    app()