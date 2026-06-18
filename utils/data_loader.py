import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict, Any

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@st.cache_data
def load_data(filepath: str = "heart.csv") -> Optional[pd.DataFrame]:
    """Loads the dataset, handles errors, and returns a dataframe."""
    try:
        if not os.path.exists(filepath):
            st.error(f"Dataset not found at {filepath}.")
            return None
            
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        st.error(f"Error loading dataset: {str(e)}")
        return None

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detects the target column ('target' or 'Target' or 'class')."""
    if df is None:
        return None
        
    candidates = ['target', 'Target', 'TARGET', 'class', 'Class', 'disease', 'HeartDisease']
    for col in candidates:
        if col in df.columns:
            return col
            
    # Fallback to the last column if it is binary
    last_col = df.columns[-1]
    if df[last_col].nunique() == 2:
        return last_col
        
    return None

@st.cache_data
def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes the dataframe and returns a quality report."""
    if df is None:
        return {}
        
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_cols": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return report

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic cleaning like dropping duplicates and handling missing values for EDA."""
    if df is None:
        return df
        
    clean_df = df.copy()
    
    # Remove duplicates
    if clean_df.duplicated().sum() > 0:
        clean_df = clean_df.drop_duplicates()
        
    # Handle missing values - simple imputation for now (median for numeric)
    num_cols = clean_df.select_dtypes(include=[np.number]).columns
    clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].median())
    
    return clean_df
