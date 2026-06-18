"""
Enhanced Heart Disease Prediction Model
Production-grade machine learning pipeline with:
- Clinical feature engineering
- Hyperparameter optimization
- Probability calibration
- Cross-validation
- Outlier detection
- SHAP explainability
- Model persistence
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import logging
from typing import Tuple, Dict, Any, List, Optional

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Configure logging
logger = logging.getLogger(__name__)


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clinical feature engineering for heart disease prediction.
    
    Creates derived features that reflect clinical knowledge:
    - Age risk groups (clinical stratification)
    - Cholesterol risk groups (lipid management)
    - Blood pressure risk groups (hypertension staging)
    - Max heart rate ratio (cardiac reserve)
    - Cholesterol-age interaction (plaque burden)
    - BP-age interaction (vascular stiffness)
    
    Args:
        df: Input DataFrame with clinical features
        
    Returns:
        DataFrame with added clinical features
    """
    df_feat = df.copy()
    
    # 1. Age Risk Group (Clinical Age Categories)
    if 'age' in df_feat.columns:
        df_feat['age_risk_group'] = (
            pd.cut(
                df_feat['age'],
                bins=[0, 45, 60, 120],
                labels=[0, 1, 2],
                include_lowest=True
            )
            .astype("int64")
        )
        
    # 2. Cholesterol Risk Group (ATP III Guidelines)
    if 'chol' in df_feat.columns:
        df_feat['cholesterol_risk_group'] = (
            pd.cut(
                df_feat['chol'],
                bins=[0, 200, 240, 1000],
                labels=[0, 1, 2],
                include_lowest=True
            )
            .astype("int64")
        )
        
    # 3. Blood Pressure Risk Group (JNC 7/8 Guidelines)
    if 'trestbps' in df_feat.columns:
        df_feat['blood_pressure_risk_group'] = (
            pd.cut(
                df_feat['trestbps'],
                bins=[0, 120, 140, 400],
                labels=[0, 1, 2],
                include_lowest=True
            )
            .astype("int64")
        )
        
    # 4. Max Heart Rate Ratio (Cardiac Reserve Indicator)
    if 'thalach' in df_feat.columns and 'age' in df_feat.columns:
        # Tanaka formula: 208 - 0.7 * age (more accurate than 220 - age)
        df_feat['max_heart_rate_ratio'] = df_feat['thalach'] / (208 - 0.7 * df_feat['age'])
        
    # 5. Cholesterol-Age Interaction (Plaque Burden Proxy)
    if 'chol' in df_feat.columns and 'age' in df_feat.columns:
        df_feat['cholesterol_age_interaction'] = df_feat['chol'] * df_feat['age'] / 100
        
    # 6. BP-Age Interaction (Vascular Stiffness Proxy)
    if 'trestbps' in df_feat.columns and 'age' in df_feat.columns:
        df_feat['bp_age_interaction'] = df_feat['trestbps'] * df_feat['age'] / 100
        
    return df_feat


def remove_outliers(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Detect and remove outliers using Isolation Forest.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        DataFrame with outliers removed
    """
    iso = IsolationForest(contamination=0.05, random_state=42)
    X = df.drop(columns=[target_col])
    outliers = iso.fit_predict(X)
    return df[outliers != -1].reset_index(drop=True)


def _evaluate_roc(df: pd.DataFrame, target_col: str) -> float:
    """
    Quick evaluator to test if features/outliers improve baseline model.
    
    Args:
        df: DataFrame to evaluate
        target_col: Target column name
        
    Returns:
        ROC-AUC score
    """
    if df.empty or len(df[target_col].unique()) < 2:
        return 0.5
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob)


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, List[str]]:
    """
    Enhanced data preparation with conditional feature engineering and outlier removal.
    
    Pipeline:
    1. Evaluate baseline performance
    2. Try clinical feature engineering
    3. Keep features if they improve ROC-AUC
    4. Try outlier removal
    5. Keep cleaned data if it improves ROC-AUC
    6. Scale features
    7. Apply SMOTE if class imbalance detected
    8. Stratified train-test split
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    
    # 1. Evaluate baseline
    baseline_auc = _evaluate_roc(df, target_col)
    logger.info(f"Baseline AUC: {baseline_auc:.4f}")
    
    # 2. Try Feature Engineering
    df_feat = add_clinical_features(df)
    feat_auc = _evaluate_roc(df_feat, target_col)
    
    if feat_auc > baseline_auc:
        logger.info(f"Feature engineering improved AUC ({baseline_auc:.4f} -> {feat_auc:.4f})")
        logger.info("Keeping engineered features.")
        df_best = df_feat
        best_auc = feat_auc
    else:
        logger.info(f"Feature engineering did not improve AUC ({baseline_auc:.4f} -> {feat_auc:.4f})")
        logger.info("Discarding engineered features.")
        df_best = df
        best_auc = baseline_auc
        
    # 3. Try Outlier Removal on the best dataset so far
    df_clean = remove_outliers(df_best, target_col)
    clean_auc = _evaluate_roc(df_clean, target_col)
    
    if clean_auc > best_auc:
        logger.info(f"Outlier removal improved AUC ({best_auc:.4f} -> {clean_auc:.4f})")
        logger.info("Keeping cleaned data.")
        df_final = df_clean
    else:
        logger.info(f"Outlier removal did not improve AUC ({best_auc:.4f} -> {clean_auc:.4f})")
        logger.info("Discarding cleaned data.")
        df_final = df_best
        
    # 4. Standard prep
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]
    
    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 6. Check for class imbalance
    class_counts = y.value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    
    if imbalance_ratio < 0.5:
        logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.3f})")
        logger.info("Applying SMOTE oversampling.")
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_scaled, y)
    else:
        X_res, y_res = X_scaled, y
        
    # 7. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
    )
    
    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Test set size: {len(X_test)} samples")
    logger.info(f"Feature count: {len(X.columns)}")
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


def _optimize_model(
    model: Any,
    param_grid: Dict[str, List],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Any:
    """
    Perform RandomizedSearchCV for hyperparameter optimization.
    
    Args:
        model: Base model instance
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Best estimator found by search
    """
    if not param_grid:
        model.fit(X_train, y_train)
        return model
        
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Trains, optimizes, calibrates and evaluates multiple models.
    
    Process for each model:
    1. Hyperparameter optimization (RandomizedSearchCV)
    2. 5-Fold Stratified Cross-Validation
    3. Probability calibration (CalibratedClassifierCV)
    4. Compare raw vs calibrated Brier Score
    5. Keep better calibrated model
    6. Calculate all metrics
    7. Generate error analysis
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        Dictionary with results for all models
    """
    
    base_models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric='logloss'),
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        )
    }
    
    results = {}
    
    for name, (base_model, params) in base_models.items():
        logger.info(f"Training {name}...")
        
        # 1. Hyperparameter Optimization
        best_base_model = _optimize_model(base_model, params, X_train, y_train)
        
        # 2. Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            best_base_model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        
        # 3. Probability Calibration
        calibrated_model = CalibratedClassifierCV(
            best_base_model,
            cv=3,
            method='sigmoid'
        )
        calibrated_model.fit(X_train, y_train)
        
        # 4. Compare raw vs calibrated Brier Score
        best_base_model.fit(X_train, y_train)
        raw_prob = best_base_model.predict_proba(X_test)[:, 1]
        cal_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        raw_brier = brier_score_loss(y_test, raw_prob)
        cal_brier = brier_score_loss(y_test, cal_prob)
        
        if cal_brier < raw_brier:
            final_model = calibrated_model
            final_prob = cal_prob
            is_calibrated = True
            logger.info(f"  Calibration improved Brier Score ({raw_brier:.4f} -> {cal_brier:.4f})")
        else:
            final_model = best_base_model
            final_prob = raw_prob
            is_calibrated = False
            logger.info(f"  Calibration did not improve Brier Score ({raw_brier:.4f} -> {cal_brier:.4f})")
            
        # 5. Final predictions
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        
        # 6. Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, final_prob)
        
        # 7. Custom score for healthcare (prioritizing recall)
        custom_score = (
            0.40 * rec +
            0.35 * roc_auc +
            0.20 * f1 +
            0.05 * prec
        )
        
        # 8. ROC curve data
        fpr, tpr, _ = roc_curve(y_test, final_prob)
        
        # 9. Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # 10. Calibration curve data
        prob_true, prob_pred = calibration_curve(y_test, final_prob, n_bins=10)
        
        # 11. Precision-Recall curve data
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, final_prob)
        
        # 12. Feature importance if available
        importance = None
        if hasattr(best_base_model, "feature_importances_"):
            importance = best_base_model.feature_importances_.tolist()
        elif hasattr(best_base_model, "coef_"):
            importance = np.abs(best_base_model.coef_[0]).tolist()
            
        # 13. Error analysis - CORRECTLY identify indices
        false_pos_indices = np.where((y_pred == 1) & (y_test == 0))[0].tolist()
        false_neg_indices = np.where((y_pred == 0) & (y_test == 1))[0].tolist()
        
        # VERIFY: These counts should match confusion matrix
        # FP count = len(false_pos_indices) should equal cm[0][1]
        # FN count = len(false_neg_indices) should equal cm[1][0]
        
        logger.info(f"  FP count: {len(false_pos_indices)}, FN count: {len(false_neg_indices)}")
        logger.info(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        
        results[name] = {
            "model_obj": final_model,
            "is_calibrated": is_calibrated,
            "metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Brier_Score": min(raw_brier, cal_brier),
                "CV_ROC_AUC_Mean": float(np.mean(cv_scores)),
                "CV_ROC_AUC_STD": float(np.std(cv_scores)),
                "Custom_Score": custom_score
            },
            "roc_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "calib_data": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
            "pr_data": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()},
            "confusion_matrix": cm,
            "feature_importance": importance,
            "false_positives": false_pos_indices,  # ✅ Store actual indices
            "false_negatives": false_neg_indices,   # ✅ Store actual indices
            "y_test_indices": list(range(len(y_test)))
        }
        
        logger.info(f"  {name} - ROC-AUC: {roc_auc:.4f}, Recall: {rec:.4f}, CV: {np.mean(cv_scores):.4f}")
        
    return results


def get_best_model(results: Dict[str, Any]) -> Tuple[str, Any, Dict]:
    """
    Selects the best model based on Custom_Score.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        best_model_name, best_model_object, best_model_metrics
    """
    best_name = None
    best_score = -1
    
    for name, data in results.items():
        score = data["metrics"]["Custom_Score"]
        if score > best_score:
            best_score = score
            best_name = name
            
    return best_name, results[best_name]["model_obj"], results[best_name]["metrics"]


def save_model(
    model_name: str,
    model_obj: Any,
    scaler: Any,
    metrics: Dict,
    features: List[str],
    all_results: Dict[str, Any] = None,
    X_train: pd.DataFrame = None,
    metadata_path: str = "models/model_metadata.json"
) -> None:
    """
    Saves the enhanced model with all associated data.
    
    Saves:
    - Best model (.joblib) using model name with underscores
    - Scaler (.joblib)
    - Feature names (.joblib)
    - Training data (.joblib) - for SHAP
    - Metadata with all results (.json)
    
    Args:
        model_name: Name of the best model
        model_obj: Trained model object
        scaler: Fitted StandardScaler
        metrics: Model metrics
        features: Feature names
        all_results: Results for all models
        X_train: Training data (for SHAP and metadata)
        metadata_path: Path for metadata JSON
    """
    os.makedirs("models", exist_ok=True)
    
    # Save model with standardized naming (spaces replaced with underscores)
    model_filename = f"models/{model_name.replace(' ', '_')}.joblib"
    joblib.dump(model_obj, model_filename)
    logger.info(f"Model saved to: {model_filename}")
    
    # Save scaler
    joblib.dump(scaler, "models/scaler.joblib")
    
    # Save feature names
    joblib.dump(list(features), "models/feature_names.joblib")
    
    # Save training data for SHAP
    if X_train is not None:
        joblib.dump(X_train, "models/X_train_scaled.joblib")
        training_samples = len(X_train)
    else:
        training_samples = 0
        logger.warning("X_train not provided. Training samples set to 0.")
    
    # Prepare serializable results with error analysis data
    serializable_results = {}
    if all_results:
        for name, data in all_results.items():
            # Ensure false_positives and false_negatives are properly serialized
            fp_list = data.get("false_positives", [])
            fn_list = data.get("false_negatives", [])
            
            serializable_results[name] = {
                "is_calibrated": data.get("is_calibrated", False),
                "metrics": data["metrics"],
                "roc_data": data["roc_data"],
                "calib_data": data.get("calib_data"),
                "pr_data": data.get("pr_data"),
                "confusion_matrix": data["confusion_matrix"],
                "feature_importance": data["feature_importance"],
                "false_positives": fp_list if fp_list is not None else [],
                "false_negatives": fn_list if fn_list is not None else []
            }
            
            # Log verification
            cm = data.get("confusion_matrix", [])
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                logger.info(f"  {name} - Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
                logger.info(f"  {name} - False Positives saved: {len(fp_list)}, False Negatives saved: {len(fn_list)}")
    
    # Build metadata with enhanced fields
    metadata = {
        "best_model": model_name,
        "model_type": str(type(model_obj).__name__),
        "training_samples": training_samples,
        "metrics": metrics,
        "features": list(features),
        "timestamp": pd.Timestamp.now().isoformat(),
        "all_results": serializable_results
    }
    
    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"✅ Model saved successfully to models/")
    logger.info(f"  Best model: {model_name}")
    logger.info(f"  Model Type: {type(model_obj).__name__}")
    logger.info(f"  Training Samples: {training_samples}")
    logger.info(f"  ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
    logger.info(f"  Recall: {metrics.get('Recall', 'N/A')}")


def load_model() -> Tuple[Any, Any, List[str], Dict]:
    """
    Load the saved model and associated data dynamically.
    
    This function reads the metadata to determine which model was saved as
    the best model, then loads that specific model file.
    
    Returns:
        model: Trained model object
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        metadata: Dictionary with all model metadata
    """
    # Load metadata
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Get the best model name from metadata
    best_model_name = metadata["best_model"]
    
    # Construct the correct filename (spaces replaced with underscores)
    model_filename = f"models/{best_model_name.replace(' ', '_')}.joblib"
    
    # Load the model
    model = joblib.load(model_filename)
    
    # Load scaler
    scaler = joblib.load("models/scaler.joblib")
    
    # Load feature names
    feature_names = joblib.load("models/feature_names.joblib")
    
    logger.info(f"✅ Loaded model: {best_model_name}")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Features: {len(feature_names)}")
    
    return model, scaler, feature_names, metadata


def predict_risk(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    input_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Make a prediction with explanation.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        input_data: Input DataFrame with patient data
        
    Returns:
        Dictionary with prediction results
    """
    # Validate input columns
    missing = set(feature_names) - set(input_data.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Ensure correct column order
    input_data = input_data[feature_names]
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Get prediction and probability
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Risk classification
    if probability < 0.30:
        risk_level = "Low Risk"
        risk_color = "#2E7D32"  # Green
    elif probability < 0.60:
        risk_level = "Moderate Risk"
        risk_color = "#F57C00"  # Orange
    else:
        risk_level = "High Risk"
        risk_color = "#C62828"  # Red
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "prediction_text": "Heart Disease Detected" if prediction == 1 else "No Significant Risk"
    }


# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv('heart.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(
        df, target_col='target'
    )
    
    # Train models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Get best model
    best_name, best_model, best_metrics = get_best_model(results)
    
    # Save model with X_train
    save_model(
        model_name=best_name,
        model_obj=best_model,
        scaler=scaler,
        metrics=best_metrics,
        features=features,
        all_results=results,
        X_train=X_train  # ✅ CRITICAL: Pass X_train here
    )
    
    # Verify saved metadata
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    print("\n✅ Verification:")
    print(f"  Best Model: {metadata['best_model']}")
    print(f"  Model Type: {metadata['model_type']}")
    print(f"  Training Samples: {metadata['training_samples']}")
    print(f"  ROC-AUC: {metadata['metrics']['ROC-AUC']}")
    
    # Verify error analysis data
    best_results = metadata['all_results'].get(best_name, {})
    fp_count = len(best_results.get('false_positives', []))
    fn_count = len(best_results.get('false_negatives', []))
    cm = best_results.get('confusion_matrix', [])
    
    if cm and len(cm) == 2:
        print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        print(f"  False Positives stored: {fp_count} (should match FP={cm[0][1]})")
        print(f"  False Negatives stored: {fn_count} (should match FN={cm[1][0]})")
        
        if fp_count == cm[0][1] and fn_count == cm[1][0]:
            print("  ✅ Error analysis data matches confusion matrix!")
        else:
            print("  ❌ Error analysis data does NOT match confusion matrix!")