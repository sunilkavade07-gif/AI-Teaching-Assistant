import streamlit as st
from google.oauth2 import service_account


def get_google_credentials():
    """
    Get Google Cloud credentials from Streamlit Secrets
    when running on Streamlit Cloud.

    Falls back to normal Google Application Default Credentials
    when running locally.
    """

    try:
        if "gcp_service_account" in st.secrets:
            credentials_info = dict(st.secrets["gcp_service_account"])

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info
            )

            return credentials

    except Exception as e:
        print(f"Streamlit Google credentials not available: {e}")

    # Local development:
    # Google Cloud SDK / Application Default Credentials
    # will be used automatically.
    return None