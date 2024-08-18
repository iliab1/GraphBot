import streamlit as st
from auth import login, login_sidebar_button

login_sidebar_button()

# Redirects to login page if user is not logged in
if "cookies" not in st.session_state:
    login()


st.title('GraphBot🤖🔗📚')
#st.subheader('Ground LLM in Graph')

# Introduction to GraphBot
st.markdown('''
Whether you're looking to extract key information, perform detailed analyses,
or gain new insights, GraphBot is here to assist in seconds 🚀
''')

if st.button('Upload your first document today🌟'):
    st.switch_page("pages/1_📁_Document_Upload.py")

# Key Features
st.subheader('Key Features')
st.markdown('''
- **📊 Knowledge Graphs:**
  Convert unstructured PDF documents into insightful knowledge graphs.
- **💬 Chat with Graph:**
  Ask questions about the document and get instant answers.
- **🔍 Multi-hop Reasoning:**
  Perform complex analysis to uncover the deepest of insights.
- **📄 Graph-Grounding:**
  Get reliable and precise information, grounded in the knowledge graph.
- **☁️ Store Knowledge:**
  Store your graphs in the Neo4j Aura cloud and access them from anywhere.
''')



