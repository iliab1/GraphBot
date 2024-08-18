import streamlit as st
import requests
from pydantic import BaseModel
from streamlit import session_state
import time
from requests.exceptions import ConnectionError, HTTPError, Timeout
from typing import Dict


# Add https://discuss.streamlit.io/t/redirecting-logger-output-to-a-streamlit-widget/61070/2

class ClientAuth(BaseModel):
    uri: str
    userName: str
    password: str
    database: str
    openai_key: str


def initialize_routes():
    """
    Initialize routes by making a request to the specified URL.
    """
    if st.session_state.get('routes_initialized', False):
        #st.info("Routes already initialized")
        return

    url = "http://backend:8080/initialize_routes"
    try:
        response = requests.get(url, cookies=st.session_state.cookies)
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("status") == "success":
            st.session_state.routes_initialized = True
            st.success(response_data.get("message", "Routes initialized successfully"))
        else:
            if "A runnable already exists at path:" in str(response_data.get('error')):
                st.session_state.routes_initialized = True
                st.warning("Routes were already initialized previously")
            else:
                st.error(f"{response_data.get('message')} - {response_data.get('error')}")
                st.stop()

    except requests.exceptions.RequestException as e:
        st.error(f"{e}")
        st.stop()  # Stop the app so the chatbot doesn't run


@st.dialog("Authentication", width="large")
def login():
    st.write("Please enter your Neo4j credentials:")
    # Login fields
    uri = st.text_input("URI", "bolt://neo4j:7687")
    username = st.text_input("User Name", "neo4j")
    password = st.text_input("Password", "pleaseletmein", type="password")
    database = st.text_input("Database Name", "neo4j")
    st.write("Please enter your OpenAI API Key:")
    openai_key = st.text_input("OpenAI API Key", type="password")

    # Login button
    if st.button("Connect"):
        # Check if all fields are filled
        if not uri or not username or not password or not database or not openai_key:
            st.error("All fields are required. Please fill in all the details.")
            st.stop()
        # Store the credentials
        user_credentials = ClientAuth(uri=uri, userName=username, password=password, database=database,
                                      openai_key=openai_key)
        data = user_credentials.dict()

        try:
            response = requests.post("http://backend:8080/set_auth_details", json=data, timeout=5)
            response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx

            response_data = response.json()

            if response_data.get("status") == "success":
                # Store cookies in session_state
                st.session_state.cookies = response.cookies

                # Success message
                st.success("Authentication details set successfully")
                time.sleep(2)

                # Rerun the app to display the main page
                st.rerun()
            else:
                st.error(f"Error setting authentication details: {response_data.get('error')}")

        except ConnectionError:
            st.error("Cannot connect to the server. Please ensure the backend is running and try again.")

        except HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")

        except Timeout:
            st.error("The request timed out. Please try again later.")

        except Exception as err:
            st.error(f"An unexpected error occurred: {err}")


def login_sidebar_button():
    if st.sidebar.button("Re-Login"):
        login()
