import streamlit as st
import sys
import os

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.theme import PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, BACKGROUND_COLOR, CARD_BACKGROUND, TEXT_COLOR

def apply_enterprise_theme():
    """Injects custom CSS to match Epic/Cerner enterprise healthcare UI."""
    st.markdown(f"""
        <style>
            /* Global Background and Fonts */
            .stApp {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            /* Sidebar Styling */
            .css-1d391kg, .css-1lcbmhc {{
                background-color: {CARD_BACKGROUND};
                border-right: 1px solid #E0E6ED;
            }}
            
            /* Metric Cards */
            div[data-testid="stMetricValue"] {{
                color: {PRIMARY_COLOR};
                font-size: 2rem;
                font-weight: 700;
            }}
            
            div[data-testid="stMetricLabel"] {{
                color: #5C6E80;
                font-size: 1rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            /* Primary Buttons */
            .stButton > button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 4px;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 600;
                transition: all 0.3s;
            }}
            .stButton > button:hover {{
                background-color: {SECONDARY_COLOR};
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            /* Secondary/Outline Buttons - targeting specific Streamlit button styles if possible */
            
            /* Card-like containers for standard elements */
            .css-1v0mbdj > .e1tzin5v2 {{
                background-color: {CARD_BACKGROUND};
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #E0E6ED;
                margin-bottom: 1rem;
            }}
            
            /* Headers */
            h1, h2, h3 {{
                color: {TEXT_COLOR};
                font-weight: 600;
            }}
            h1 {{
                border-bottom: 2px solid {PRIMARY_COLOR};
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem;
            }}
            
            /* Dataframes */
            .dataframe {{
                font-size: 0.9rem;
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                background-color: {CARD_BACKGROUND};
                color: {TEXT_COLOR};
                font-weight: 600;
                border: 1px solid #E0E6ED;
                border-radius: 4px;
            }}
        </style>
    """, unsafe_allow_html=True)

def page_header(title: str, subtitle: str = ""):
    """Renders a standardized page header."""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: #7F8C8D; font-size: 1.1rem; margin-top: -1rem; margin-bottom: 2rem;'>{subtitle}</p>", unsafe_allow_html=True)

def metric_card(title: str, value: str, delta: str = None):
    """Renders a styled metric using Streamlit's native component."""
    st.metric(label=title, value=value, delta=delta)
