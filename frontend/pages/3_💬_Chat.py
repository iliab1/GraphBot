import streamlit as st
from langserve import RemoteRunnable
from langchain_core.messages import AIMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx, get_script_run_ctx
from time import sleep
import requests
from auth import login, initialize_routes, login_sidebar_button

st.set_page_config(layout="wide")
login_sidebar_button()

if "cookies" not in st.session_state:
    login()
    st.stop()

st.title("Multi-Response Chatbot")
st.write("This chatbot can respond using different retrival strategies. "
         "Select the strategies you want to use.")


# Check if routes have been initialized
initialize_routes()


# From https://discuss.streamlit.io/t/multiple-simultaneous-write-stream-elements-running-at-the-same-time/69223/2
MODEL_CHOICES = ["Naive_RAG", "Parent_RAG", "Neighbour_RAG", "Hybrid_RAG"]
selected_models = st.multiselect("Select strategies:", MODEL_CHOICES, default=MODEL_CHOICES)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how can I help you?"),
    ]


def generate_data(response_stream):
    """A generator to pass to st.write_stream"""
    try:
        buffer = ""
        for response in response_stream:
            buffer += response
            while buffer:
                space_index = buffer.find(" ", 1) if " " in buffer else len(buffer)
                if space_index == -1:
                    break
                yield buffer[:space_index + 1]
                buffer = buffer[space_index + 1:]
                sleep(0.05)
        if buffer:
            yield buffer
    except Exception as e:
        yield f"Error: {str(e)}"


def call_chain(strategy, user_query):
    typical_chain_url = "http://backend:8080/typical_chain"
    parent_chain_url = "http://backend:8080/parent_chain"
    graph_neighbour_url = "http://backend:8080/graph_chain"
    advanced_chain_url = "http://backend:8080/advanced_chain"

    # Logic to switch between different chains
    if strategy == "Naive_RAG":
        chain_url = typical_chain_url
    elif strategy == "Parent_RAG":
        chain_url = parent_chain_url
    elif strategy == "Neighbour_RAG":
        chain_url = graph_neighbour_url
    elif strategy == "Hybrid_RAG":
        chain_url = advanced_chain_url
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    chain = RemoteRunnable(chain_url)

    response_stream = chain.stream({
        "question": user_query,
        # "chat_history": st.session_state.chat_history
    })
    #response_stream = chain.stream(chain.stream)
    return response_stream


def stream_data_in_column(column, strategy, user_query, ctx):
    """Populate columns simultaneously"""
    add_script_run_ctx(current_thread(), ctx)
    response_stream = call_chain(strategy, user_query)
    with column:
        st.write_stream(generate_data(response_stream))


def threading_output(prompt):
    ctx = get_script_run_ctx()
    if len(selected_models) == 1:
        cols = [st]
    else:
        cols = st.columns(len(selected_models))

    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = [
            executor.submit(stream_data_in_column, col, strategy, prompt, ctx) for col, strategy in
            zip(cols, selected_models)
        ]


# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    threading_output(user_query)
