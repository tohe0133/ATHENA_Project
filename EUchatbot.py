import streamlit as st
from streamlit_chat import message
import requests

# Retrieve the OpenAI API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]


def callGPT():
    # API endpoint and headers
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload for GPT API
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": st.session_state.conversation,
        "temperature": 0.7
    }

    # Send a request to the GPT API
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        # Extract the chatbot's response content
        chatbot_response = response.json()["choices"][0]["message"]["content"]

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🇪🇺"):
            st.markdown(chatbot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "avatar": "🇪🇺", "content": chatbot_response})
        st.session_state.conversation.append({"role": "assistant", "content": chatbot_response})
    else:
        st.markdown("There was an error with status code: " + str(response.status_code))
        if response.status_code == 400:
            st.markdown("The text input was to long")
    return


# Initialize the conversation in session state or with a default value
if 'conversation' not in st.session_state:
    st.session_state.conversation = [
        {"role": "system",
         "content": "Your primary function is to provide concise summaries of EU articles and legislations based on the titles of the text or if the user inputs the full article. Your summary should be accurate and informative. Give the user some hyperlinks to the original source and to further reading to give a deeper understanding. Also give hyperlinks to give context to things that might be mentioned in the article. Once a summary is provided, remain available to answer any follow-up questions. Ensure that your responses prioritize clarity and user engagement."}
    ]
if 'setting' not in st.session_state:
    st.session_state.setting = "False"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    user_input = ""

st.title("EU-Summarizer")
st.markdown(
    "This is a chatbot that can summerize an EU aritcle or EU legislation. If you post the full text or the title of the article the chatbot will do its best to make a good summary. Afterwards you can ask questions about it or type in 'Quiz' to get a quiz based on the article.")

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Display chat messages from history on app rerun and applies the correct icon
for message in st.session_state.messages:
    is_user = message["role"] == "user"
    if is_user:
        with st.chat_message(message["role"], avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="🇪🇺"):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter text"):
    user_input = prompt
    # Display user message in chat message container
    st.chat_message("user", avatar="👤").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if user_input.lower() == "quiz":
    setting = user_input.lower()
    st.session_state.conversation.append({"role": "system", "content": "Give the user a three question multiple-choice quiz, let them answer and then evaluate if they are correct"})
    with st.spinner("Loading..."):
        callGPT()
    st.session_state.setting = "waiting"
elif prompt:
    st.session_state.conversation.append({"role": "user", "content": prompt})

    # Clear the input box by setting it to an empty string
    prompt = ""

    with st.spinner("Loading..."):
        callGPT()
    if st.session_state.setting == "False":
        with st.chat_message("assistant", avatar="🇪🇺"):
            st.markdown(
                "Would you like to ask any more questions about the text or would you like a short quiz to check if you've understood the text? (Input 'Quiz' if you want a quiz)")
    st.session_state.setting = "waiting"