# ANTI-GRAVITY EXECUTION DIRECTIVE

You are not building a demo.

You are building a deployable healthcare analytics SaaS application.

Priority Order:

1. Correctness
2. Reliability
3. Medical Interpretability
4. Performance
5. UX
6. Visual Design

Never create placeholder functionality.

Every button must perform a real action.

Every chart must use real data.

Every metric must be computed from heart.csv.

Every prediction must come from a trained model.

Every report must be generated dynamically.

Every insight must be derived from actual dataset statistics.

Never fabricate:
- Accuracy
- ROC-AUC
- Feature Importance
- Risk Scores
- Dataset Metrics

If a value cannot be calculated:
display an explanation instead.

--------------------------------------------------

# REQUIRED ARCHITECTURE

Use:

Backend:
- Python

Framework:
- Streamlit

ML:
- Scikit-Learn
- Imbalanced-Learn

Visualization:
- Plotly

Reporting:
- ReportLab

Storage:
- Joblib
- JSON

Project Structure:

heart_disease_app/

├── app.py
├── heart.csv
├── pages/
├── utils/
├── models/
├── reports/
├── assets/
├── logs/
├── config/
├── requirements.txt

--------------------------------------------------

# MANDATORY MACHINE LEARNING WORKFLOW

Load Dataset
→ Validate Schema
→ Validate Values
→ Detect Missing Values
→ Remove Duplicates
→ EDA
→ Feature Engineering
→ Train/Test Split
→ Class Imbalance Check
→ SMOTE if Needed
→ Hyperparameter Search
→ Model Comparison
→ Save Best Model
→ Generate Dashboard
→ Enable Predictions

--------------------------------------------------

# MODEL SELECTION RULE

Best model must be selected using:

1. ROC-AUC
2. Recall
3. F1 Score

NOT Accuracy Alone.

Healthcare applications prioritize Recall.

--------------------------------------------------

# ENTERPRISE UI REQUIREMENTS

The interface must look comparable to:

- Epic Systems
- Cerner
- Philips HealthSuite
- IBM Watson Health

Use:

Primary:
#1E88E5

Secondary:
#4DA6FF

Accent:
#E53935

Background:
#F5F9FC

Cards:
#FFFFFF

Design Style:

- Professional
- Clinical
- Clean
- Premium
- Enterprise

No childish gradients.

No neon colors.

No gaming-style UI.

--------------------------------------------------

# PERFORMANCE REQUIREMENTS

Use caching everywhere appropriate.

@st.cache_data
@st.cache_resource

Training should only rerun when:

- Dataset changes
- User clicks Retrain

--------------------------------------------------

# REPORT QUALITY

Generated PDF must include:

- Cover Page
- Executive Summary
- Dataset Overview
- EDA Findings
- Model Performance
- ROC Curve
- Feature Importance
- Prediction Results
- Recommendations
- Disclaimer

Professional formatting required.

--------------------------------------------------

# SECURITY

Validate all inputs.

Validate all file operations.

Prevent crashes from malformed datasets.

Store logs in:

logs/errors.log

Store model metadata in:

models/model_metadata.json

--------------------------------------------------

# FINAL EXPECTATION

Produce a production-grade Heart Disease Risk Prediction Platform that appears ready for:

- Hospital Demonstrations
- Startup MVP Launch
- Healthcare Analytics Portfolio
- Internship Showcase
- Client Presentation

The application should feel like a commercial software product worth selling rather than an academic project.