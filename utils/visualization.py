import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
from config.theme import PLOTLY_TEMPLATE, COLOR_SCALE, PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, BACKGROUND_COLOR

def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Applies enterprise theme to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Segoe UI", color="#2C3E50"),
        title_font=dict(size=18, family="Segoe UI", color="#2C3E50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_target_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    value_counts = df[target_col].value_counts().reset_index()
    value_counts.columns = [target_col, 'Count']
    if len(value_counts) == 2:
        value_counts[target_col] = value_counts[target_col].astype(str).map({'0': 'Negative (0)', '1': 'Positive (1)'})
    fig = px.bar(
        value_counts, x=target_col, y='Count', color=target_col,
        color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR],
        title=f"Distribution of {target_col}"
    )
    return apply_chart_theme(fig)

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include=['number']).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu_r', zmin=-1, zmax=1
    ))
    fig.update_layout(title="Feature Correlation Matrix")
    return apply_chart_theme(fig)

def plot_roc_curve(fpr, tpr, roc_auc, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC = {roc_auc:.3f})", 
                             mode='lines', line=dict(color=PRIMARY_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", 
                             mode='lines', line=dict(color='gray', dash='dash')))
    fig.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return apply_chart_theme(fig)

def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance_df.sort_values(by='Importance', ascending=True),
        x='Importance', y='Feature', orientation='h', color='Importance',
        color_continuous_scale=[SECONDARY_COLOR, PRIMARY_COLOR],
        title="Global Feature Importance"
    )
    return apply_chart_theme(fig)

def plot_calibration_curve(prob_true, prob_pred, model_name: str) -> go.Figure:
    """Plots probability calibration curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name=model_name, line=dict(color=PRIMARY_COLOR)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated', line=dict(color='gray', dash='dash')))
    fig.update_layout(title="Calibration Curve (Reliability Diagram)", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives")
    return apply_chart_theme(fig)

def plot_precision_recall_curve(precision, recall, model_name: str) -> go.Figure:
    """Plots Precision-Recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=model_name, line=dict(color=ACCENT_COLOR)))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return apply_chart_theme(fig)

def plot_shap_summary(model, X_train_scaled, features):
    """Generates SHAP summary using matplotlib since SHAP plotting with Plotly is complex natively."""
    import matplotlib.pyplot as plt
    
    # Check if model has standard predict or is calibrated
    # For CalibratedClassifierCV or complex pipelines, use KernelExplainer or TreeExplainer if tree
    try:
        if hasattr(model, 'estimators_'): # It's an ensemble or CalibratedClassifierCV
            if hasattr(model.estimator, 'feature_importances_'):
                explainer = shap.TreeExplainer(model.estimator)
                shap_values = explainer.shap_values(X_train_scaled)
            else:
                explainer = shap.LinearExplainer(model.estimator, X_train_scaled)
                shap_values = explainer.shap_values(X_train_scaled)
        elif hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_scaled)
        else:
            explainer = shap.LinearExplainer(model, X_train_scaled)
            shap_values = explainer.shap_values(X_train_scaled)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle multi-class output format in shap
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
            
        shap.summary_plot(shap_values_to_plot, X_train_scaled, feature_names=features, show=False)
        return fig, explainer, shap_values
    except Exception as e:
        return None, None, None

def plot_shap_waterfall(explainer, shap_values, instance_idx, features, input_scaled=None):
    """Generates SHAP waterfall plot for a single instance."""
    import matplotlib.pyplot as plt
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Check SHAP version / format
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            expected_value = 0
            
        if isinstance(shap_values, list):
            sv = shap_values[1][instance_idx]
        else:
            sv = shap_values[instance_idx] if len(shap_values.shape) > 1 else shap_values
            
        shap.waterfall_plot(shap.Explanation(values=sv, base_values=expected_value, data=input_scaled[0] if input_scaled is not None else None, feature_names=features), show=False)
        return fig
    except Exception as e:
        return None
