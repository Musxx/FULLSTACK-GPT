import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 하는 역할 : Streamlit 페이지 기본 설정
st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("❓ QuizGPT")
st.write("난이도를 선택하고 시험을 치르세요.")


# 하는 역할 : 사이드바 UI 구성 (API Key, 난이도, 출처 선택)
with st.sidebar:
    st.header("🔑 OpenAI 설정")

    # 하는 역할 : 유저가 직접 OpenAI API Key 입력
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    st.divider()

    # 하는 역할 : 시험 난이도 선택 (LLM 프롬프트에 전달)
    difficulty = st.selectbox(
        "시험 난이도",
        ("Easy", "Medium", "Hard")
    )

    st.divider()

    # 하는 역할 : 문제 출처 선택 (파일 or 위키피디아)
    source = st.selectbox(
        "문제 출처",
        ("File", "Wikipedia")
    )

    st.divider()

    # 하는 역할 : GitHub 리포지토리 링크 표시
    st.markdown(
        "🔗 **GitHub Repository**  \n"
        "[https://github.com/yourname/quizgpt](https://github.com/yourname/quizgpt)"
    )


# 하는 역할 : API Key 미입력 시 실행 중단
if not openai_api_key:
    st.warning("OpenAI API Key를 입력해주세요.")
    st.stop()


# 하는 역할 : OpenAI LLM 객체 생성
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)


# 하는 역할 : Function Calling으로 퀴즈 JSON 구조 강제
quiz_function = {
    "name": "create_quiz",
    "description": "난이도와 언어에 맞는 객관식 퀴즈 생성",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"}
                                },
                                "required": ["answer", "correct"]
                            }
                        }
                    },
                    "required": ["question", "answers"]
                }
            }
        },
        "required": ["questions"]
    }
}


# 하는 역할 : 입력 문서의 언어 감지 (ISO 639-1 코드)
def detect_language(text: str) -> str:
    prompt = f"""
다음 텍스트의 주된 언어를 ISO 639-1 코드로만 반환하세요.

예:
한국어 → ko
영어 → en
일본어 → ja

텍스트:
{text[:1500]}
"""
    result = llm.predict(prompt)
    return result.strip().lower()

# 하는 역할 : 언어 코드 → 사용자 친화적 언어명 매핑
LANG_MAP = {
    "ko": "한국어",
    "en": "영어",
    "ja": "일본어"
}


# 하는 역할 : 업로드된 파일을 텍스트로 변환하고 분할
def load_and_split(file):
    os.makedirs("./.cache", exist_ok=True)
    path = f"./.cache/{file.name}"

    # 하는 역할 : 업로드 파일을 로컬에 저장
    with open(path, "wb") as f:
        f.write(file.getvalue())

    # 하는 역할 : 문서 로딩 및 텍스트 분할
    loader = UnstructuredFileLoader(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = loader.load_and_split(splitter)

    # 하는 역할 : 여러 Document를 하나의 문자열로 결합
    return "\n\n".join(d.page_content for d in docs)

# 하는 역할 : Wikipedia에서 주제 검색 후 텍스트 추출
def wiki_search(term: str) -> str:
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(term)
    return "\n\n".join(d.page_content for d in docs)

# 하는 역할 : 문서 + 난이도 + 언어를 기반으로 퀴즈 생성
def generate_quiz(context: str, difficulty: str):
    # 하는 역할 : 문서 언어 감지
    lang_code = detect_language(context)
    lang_name = LANG_MAP.get(lang_code, "한국어")

    # 하는 역할 : LLM에게 퀴즈 생성 지시
    prompt = f"""
당신은 퀴즈 출제자입니다.

문제와 보기를 반드시 **{lang_name}**로 작성하세요.
입력 문서의 언어와 일치해야 합니다.

난이도: {difficulty}

아래 내용을 기반으로
객관식 문제 5개를 만드세요.

{context}
"""

    response = llm.predict_messages(
        [HumanMessage(content=prompt)],
        functions=[quiz_function],
        function_call={"name": "create_quiz"}
    )

    # 하는 역할 : Function Calling 결과(JSON) 파싱
    return json.loads(response.additional_kwargs["function_call"]["arguments"])

# 하는 역할 : 문서 로딩 분기 처리
docs = None

if source == "File":
    # 하는 역할 : 파일 업로드 처리
    file = st.file_uploader("파일 업로드", type=["txt", "pdf", "docx"])
    if file:
        docs = load_and_split(file)
else:
    # 하는 역할 : Wikipedia 주제 입력 처리
    topic = st.text_input("Wikipedia 주제 입력")
    if topic:
        docs = wiki_search(topic)

# 하는 역할 : 문서 미선택 시 실행 중단
if not docs:
    st.info("문서를 선택해주세요.")
    st.stop()

# 하는 역할 : 퀴즈 상태를 session_state에 저장
if "quiz" not in st.session_state:
    st.session_state.quiz = generate_quiz(docs, difficulty)
    st.session_state.submitted = False

quiz = st.session_state.quiz

# 하는 역할 : 퀴즈 응시 폼 렌더링
with st.form("quiz_form"):
    answers = []

    for idx, q in enumerate(quiz["questions"]):
        st.subheader(f"Q{idx+1}. {q['question']}")

        # 하는 역할 : 객관식 보기 선택 (초기 미선택 상태)
        choice = st.radio(
            "선택",
            [a["answer"] for a in q["answers"]],
            index=None,
            key=f"q_{idx}"
        )

        answers.append((choice, q["answers"]))

    # 하는 역할 : 시험 제출 버튼
    submitted = st.form_submit_button("제출")


# 하는 역할 : 제출 후 점수 계산 및 피드백 제공
if submitted:
    # 하는 역할 : 모든 문제 선택 여부 검증
    if any(choice is None for choice, _ in answers):
        st.warning("모든 문제를 선택해주세요.")
        st.stop()

    # 하는 역할 : 정답 채점
    score = 0
    for selected, options in answers:
        for opt in options:
            if opt["answer"] == selected and opt["correct"]:
                score += 1

    # 하는 역할 : 점수 출력
    st.success(f"점수: {score} / {len(quiz['questions'])}")

    # 하는 역할 : 만점 여부에 따른 분기 처리
    if score == len(quiz["questions"]):
        st.balloons()
        st.success("🎉 만점입니다! 축하드립니다.")
    else:
        # 하는 역할 : 재시험 허용
        if st.button("다시 시험 보기"):
            del st.session_state.quiz
            del st.session_state.submitted
            st.experimental_rerun()
