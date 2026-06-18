import streamlit as st
import pandas as pd
from utils.ui_components import page_header
from utils.data_loader import load_data, preprocess_data, detect_target_column
from utils.ml_pipeline import prepare_data, train_and_evaluate_models, get_best_model, save_model

st.set_page_config(page_title="Model Training Center", layout="wide")

def app():
    page_header("Model Training Center", "Configure, train, and automatically select the best predictive model")
    
    df = load_data()
    if df is not None:
        clean_df = preprocess_data(df)
        target_col = detect_target_column(clean_df)
        
        if not target_col:
            st.error("Could not automatically detect the target column. Cannot proceed with training.")
            return
            
        st.markdown("### Training Configuration")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        with col2:
            random_state = st.number_input("Random Seed (for reproducibility)", min_value=1, max_value=9999, value=42)
            
        st.info("The pipeline will automatically check for class imbalance and apply SMOTE if necessary. Models trained: Logistic Regression, Random Forest, XGBoost.")
        
        if st.button("🚀 Start Training Pipeline", type="primary"):
            with st.spinner("Preparing data, scaling, and handling class imbalance..."):
                X_train, X_test, y_train, y_test, scaler, features = prepare_data(clean_df, target_col, test_size=test_size, random_state=random_state)
                
            with st.spinner("Optimizing, calibrating and training models (LR, RF, XGB)..."):
                results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
                
            with st.spinner("Selecting best model prioritizing Recall and ROC-AUC..."):
                best_name, best_model, best_metrics = get_best_model(results)
                
                # Save data needed for SHAP explainer
                all_results = results
                save_model(best_name, best_model, scaler, best_metrics, features, all_results=all_results, X_train=X_train)
                
            st.success(f"Training & Optimization Complete! Best Model: **{best_name}**")
            st.markdown(f"""
            **Winning Metrics:**
            * Custom Priority Score: `{best_metrics['Custom_Score']:.4f}`
            * Recall: `{best_metrics['Recall']:.4f}`
            * ROC-AUC: `{best_metrics['ROC-AUC']:.4f}`
            * Calibrated Brier Score: `{best_metrics['Brier_Score']:.4f}`
            """)
            st.info(f"Model Calibration Status: **{'Calibrated' if results[best_name].get('is_calibrated') else 'Raw'}**")
            st.info("You can now proceed to the Model Evaluation Center or view the Model Comparison dashboard.")
            
    else:
        st.error("Dataset not available.")

if __name__ == "__main__":
    app()
