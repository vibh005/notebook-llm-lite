import os
import streamlit.components.v1 as components

BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")

firebase_component = components.declare_component(
    "firebase_auth",
    path=BUILD_DIR,
)


class FirebaseAuth:
    def __init__(self, firebase_config, lang="en"):
        self.firebase_config = firebase_config
        self.lang = lang

    def login_form(self, providers=None):
        if providers is None:
            providers = ["email", "google"]

        return firebase_component(
            name="LoginForm",
            firebase_config=self.firebase_config,
            lang=self.lang,
            height=500,
            default=None,
            providers=providers,
        )

    def logout_form(self):
        return firebase_component(
            name="LogoutForm",
            firebase_config=self.firebase_config,
            lang=self.lang,
            default=None,
        )

    def check_session(self):
        return firebase_component(
            name="CheckSession",
            firebase_config=self.firebase_config,
            lang=self.lang,
            default=None,
        )