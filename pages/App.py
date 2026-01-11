import streamlit as st
from openai import OpenAI
import time

st.set_page_config(page_title="Investor Assistant")

# --- Sidebar ---
st.sidebar.title("설정")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.markdown(
    "[📂 GitHub Repository](https://github.com/your-name/your-repo)"
)

if not api_key:
    st.warning("API Key를 입력하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Assistant 생성 (1회) ---
if "assistant" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You are an investor assistant that analyzes stocks.",
        model="gpt-4-0613"
    )
    st.session_state.assistant = assistant

# --- Thread 생성 ---
if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread = thread

st.title("📈 Investor Assistant")

# --- 채팅 기록 표시 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 유저 입력 ---
prompt = st.chat_input("질문을 입력하세요")

if prompt:
    # UI 표시
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Thread에 메시지 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=prompt
    )

    # Run 실행
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id
    )

    # Run 완료 대기
    while run.status != "completed":
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread.id,
            run_id=run.id
        )

    # Assistant 응답 가져오기
    messages = client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id
    )

    answer = messages.data[0].content[0].text.value

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
