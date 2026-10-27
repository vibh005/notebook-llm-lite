
import os
import firebase_admin
from firebase_admin import auth, credentials
from dotenv import load_dotenv

load_dotenv()


def initialize_firebase():
    """Initialize Firebase Admin SDK locally or on Streamlit Cloud."""
    import json
    import streamlit as st

    if firebase_admin._apps:
        return

    if "firebase_service_account" in st.secrets:
        service_account_info = dict(st.secrets["firebase_service_account"])
        credential = credentials.Certificate(service_account_info)
    else:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            raise RuntimeError(
                "Firebase credentials are missing."
            )

        if not os.path.isabs(credentials_path):
            credentials_path = os.path.join(os.getcwd(), credentials_path)

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Firebase credentials file not found: {credentials_path}"
            )

        credential = credentials.Certificate(credentials_path)

    firebase_admin.initialize_app(credential)


def verify_firebase_token(id_token: str) -> dict:
    """Verify a Firebase ID token and return the user's identity."""
    initialize_firebase()

    if not id_token:
        raise ValueError("Firebase ID token is required.")

    decoded_token = auth.verify_id_token(id_token)

    return {
        "uid": decoded_token["uid"],
        "email": decoded_token.get("email", ""),
        "name": decoded_token.get("name", ""),
    }
