import streamlit as st
import os
from utils.ui_components import page_header

st.set_page_config(page_title="Settings & Logs", layout="wide")

def app():
    page_header("Settings & System Logs", "System configurations and error tracking")
    
    st.markdown("### Application Information")
    st.write("**Version:** 1.0.0-production")
    st.write("**Environment:** Healthcare SaaS Production")
    
    st.markdown("---")
    st.markdown("### System Logs")
    
    log_path = "logs/errors.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = f.readlines()
            
        if logs:
            st.code("".join(logs[-50:]), language="text") # Show last 50 lines
            
            if st.button("Clear Logs"):
                open(log_path, "w").close()
                st.success("Logs cleared.")
                st.rerun()
        else:
            st.info("No errors logged. System operating normally.")
    else:
        st.info("Log file not found. System operating normally.")

if __name__ == "__main__":
    app()
