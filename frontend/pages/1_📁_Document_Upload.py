import streamlit as st
import requests
from auth import login, login_sidebar_button

login_sidebar_button()

if "cookies" not in st.session_state:
    login()
    st.stop()

st.title("Document Upload")


def upload_file(file):
    with st.spinner("Uploading file..."):
        files = {"file": (file.name, file, file.type)}
        response = requests.post(
            "http://backend:8080/uploadfile/",
            files=files,
            cookies=st.session_state.cookies
        )
        response_data = response.json()
        return response_data


uploaded_file = st.file_uploader("Choose a file you want to upload", type="pdf")

if st.button("Upload"):
    if uploaded_file:
        result = upload_file(uploaded_file)
        st.write(result.get('message'))

st.write("Click the button below to resolve entities inside the graph.")



# Resolve entities
def resolve_entities():
    with st.spinner("Resolving entities..."):
        response = requests.post(
            "http://backend:8080/perform_entity_resolution",
            cookies=st.session_state.cookies
        )
        response_data = response.json()
        return response_data


# Create button
if st.button("Resolve Entities"):
    resolve_entities()
