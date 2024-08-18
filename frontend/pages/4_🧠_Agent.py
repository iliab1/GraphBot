import streamlit as st
from langserve import RemoteRunnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import asyncio
from auth import login, initialize_routes, login_sidebar_button

st.set_page_config(layout="wide")
login_sidebar_button()

if "cookies" not in st.session_state:
    login()
    st.stop()

st.title("Agent Chatbot")
st.write("This chatbot uses advanced multi-hop strategies to answer questions. "
         "Type your message below to get started.")

# Check if routes have been initialized
initialize_routes()

# Initialise agent
agent = RemoteRunnable("http://backend:8080/qa_agent")

# Initialise chat history
msgs = StreamlitChatMessageHistory(key="langchain_messages")

#Check if state variables are initialised
if 'debug_text' not in st.session_state:
    st.session_state.debug_text = ""

if 'full_response' not in st.session_state:
    st.session_state.full_response = ""

# If no messages, add a default message
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, how can I help you?")

# Render current messages
for msg in msgs.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)


# Streaming log messages from langchain abstractions
# Langserve doesn't support streaming logs yet, so we will use a placeholder to stream the response
async def run_agent(user_query):
    # Clear the debug text and full response before starting
    st.session_state.debug_text = ""
    st.session_state.full_response = ""
    st.session_state.final_answer = ""

    # Cant nest status bars, but can use spinner to show the overall status
    with st.spinner('Running Agent...') as main_status:

        async for event in agent.astream_events(
                {"input": user_query}, version="v1"
        ):

            kind = event["event"]

            # Handle chain start event
            if kind == "on_chain_start":
                if event["name"] == "/qa_agent":
                    with st.status("Starting Chain", expanded=False) as status:
                        # Update the status
                        st.write("Starting chain...")
                        status.update(
                            label=f"Starting agent: {event['name']} with input: {event['data'].get('input')}",
                            state="running",
                            expanded=False
                        )

            # Handle chain end event
            if kind == "on_chain_end":
                if event["name"] == "/qa_agent":
                    with st.status("Ending chain", expanded=True) as status:
                        # Update the status
                        st.write("Ending chain...")
                        st.write(f"Output: {event['data'].get('output')['output']}")
                        status.update(
                            label=f"Ending agent: {event['name']} with output: {event['data'].get('output')['output']}\n",
                            state="complete",
                            expanded=False
                        )
                        # Save the final response
                        st.session_state.final_answer = event["data"].get("output")["output"]

            # Handle llm model start event
            if kind == "on_chat_model_start":
                with st.status("Starting model response", expanded=False) as status:
                    # Update the status
                    st.write("Starting model response...")
                    #st.write(f"Output: {event['data'].get('output')['output']}")
                    status.update(
                        label=f"Starting model response...\n",
                        state="running",
                        expanded=False
                    )

            # Handle llm model end event
            if kind == "on_chat_model_end":
                with st.status("Ending model response", expanded=False) as status:
                    # TypeError: 'LLMResult' object is not subscriptable
                    """
                    output = event['data']['output']
                    if output and output.get('generations'):
                        generations = output['generations']
                        first_generation = generations[0][0]  # Get the first generation
                        message_content = first_generation['message']['content']
                        st.write(f"Output: {message_content}")
                    """

                    status.update(
                        label=f"Ending chat model...\n",
                        state="complete",
                        expanded=False
                    )

            # Stream the response
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                # If content is not empty, write it
                if content:
                    #main_status.spinner('Streaming Response...')
                    st.session_state.full_response += content
                    st.session_state.full_response = st.session_state.full_response.replace('$', '&#36;')
                    message_placeholder.markdown(st.session_state.full_response, unsafe_allow_html=True)


            # Debugging for tools
            elif kind == "on_tool_start":
                with st.status(f"Starting tool{event['name']}", expanded=False) as status:
                    st.write(f"Starting tool{event['name']}")
                    status.update(
                        label=f"Starting tool: {event['name']}\n",
                        state="running",
                        expanded=False
                    )

            elif kind == "on_tool_end":
                with st.status(f"Ending tool{event['name']}", expanded=False) as status:
                    st.write(f"Ending tool{event['name']}")
                    status.update(
                        label=f"Ending tool: {event['name']}\n",
                        state="complete",
                        expanded=False
                    )


# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input("Type your message here..."):
    # Write the user input
    st.chat_message("user").write(prompt)
    msgs.add_user_message(prompt)  # Add the user's message to the history
    with st.chat_message("assistant"):
        global message_placeholder
        message_placeholder = st.empty()
        message_placeholder.markdown(
            'Please wait...',
            unsafe_allow_html=True
        )
        # Run the async function properly within a sync context
        asyncio.run(run_agent(prompt))  # TODO: Add chat history
        message_placeholder.markdown(st.session_state.full_response, unsafe_allow_html=True)

        msgs.add_ai_message(st.session_state.final_answer)  # Add the AI's message to the history
