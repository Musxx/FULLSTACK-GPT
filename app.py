import streamlit as st
from datetime import datetime

today = datetime.today().strftime('%H:%M:%S')

st.set_page_config(
    page_title="app",
    page_icon="🏠",
)

import os
import streamlit as st

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="default")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.warning("OpenAI API Key를 입력하세요.")
    st.stop()

with st.sidebar: st.markdown(
      st.markdown("[👉 프로젝트 링크](https://github.com/Musxx/FULLSTACK-GPT/commit/28bf83fb24e11d023e48c81de5d70001bcf722cb)")
) 

st.title("Here are the apps I made:")

if st.checkbox("DocumentGPT", value=True):
    st.success("좌측 사이드바에서 DocumentGPT를 선택하세요.")

if st.checkbox("PrivateGPT"):
    st.info("PrivateGPT 페이지가 준비되어 있습니다.")

if st.checkbox("QuizGPT"):
    st.info("QuizGPT 페이지가 준비되어 있습니다.")