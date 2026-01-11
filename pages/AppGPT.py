import streamlit as st
import openai

st.set_page_config(page_title="Investor Assistant")

# Sidebar
st.sidebar.title("설정")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.markdown(
    "[📂 GitHub Repository](https://github.com/Musxx/FULLSTACK-GPT)"
)

if not api_key:
    st.stop()

openai.api_key = api_key

st.title("📈 Investor Assistant")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are an investor assistant that analyzes stocks."
        }
    ]

# Display history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("질문을 입력하세요")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
