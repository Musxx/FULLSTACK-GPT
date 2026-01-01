import streamlit as st
from datetime import datetime

today = datetime.today().strftime('%H:%M:%S')

st.set_page_config(
    page_title="app",
    page_icon="🏠",
)

st.title("Here are the apps I made:")

if st.checkbox("DocumentGPT", value=True):
    st.success("좌측 사이드바에서 DocumentGPT를 선택하세요.")

if st.checkbox("PrivateGPT"):
    st.info("PrivateGPT 페이지가 준비되어 있습니다.")

if st.checkbox("QuizGPT"):
    st.info("QuizGPT 페이지가 준비되어 있습니다.")