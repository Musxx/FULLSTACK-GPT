import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# =========================
# Streamlit 설정
# =========================
st.set_page_config(page_title="SiteGPT", page_icon="🌐")
st.title("🌐 SiteGPT")
st.write("웹사이트 문서를 기반으로 질문에 답변합니다.")

# =========================
# OpenAI API Key 입력
# =========================
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("OpenAI API Key를 입력하세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
)

# =========================
# Prompt 정의
# =========================
answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 제공된 문서를 기반으로만 답변하는 한국어 AI입니다.
            문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고 답하세요.

            문서 내용:
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that are the most helpful and favor the most recent ones.
            Cite sources and return the sources exactly as provided.

            Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# =========================
# HTML 파싱 (토큰 절약)
# =========================
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return soup.get_text()

# =========================
# Retriever 생성
# =========================
@st.cache_resource(show_spinner="Loading website documents...")
def load_retriever(urls: list[str]):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )

    all_docs = []

    for url in urls:
        loader = SitemapLoader(
            url,
            parsing_function=parse_page,
            filter_urls=[
                r".*/ai/.*",
                r".*/ai-gateway/.*",
                r".*/vectorize/.*",
            ],
        )

        docs = loader.load_and_split(text_splitter=splitter)
        all_docs.extend(docs)

    vectorstore = FAISS.from_documents(
        all_docs,
        OpenAIEmbeddings(),
    )

    return vectorstore.as_retriever(search_kwargs={"k": 1})


# =========================
# 문서별 답변 생성
# =========================
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    chain = answers_prompt | llm

    answers = []
    for doc in docs:
        context = doc.page_content[:3000]  # 🔥 토큰 보호

        result = chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        answers.append(
            {
                "answer": result.content,
                "source": doc.metadata.get("source"),
                "date": doc.metadata.get("lastmod"),
            }
        )

    return {
        "question": question,
        "answers": answers,
    }

# =========================
# 최종 답변 선택
# =========================
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    condensed = "\n\n".join(
        f"{a['answer']}\nSource: {a['source']}\nDate: {a['date']}"
        for a in answers
    )

    chain = choose_prompt | llm

    return chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

# =========================
# Sidebar: Sitemap 입력
# =========================
with st.sidebar:
    urls_input = st.text_area(
        "Sitemap URLs (one per line)",
        placeholder="https://example.com/sitemap.xml",
    )

urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

if not urls:
    st.info("사이드바에 Sitemap URL을 입력하세요.")
elif not all(".xml" in u for u in urls):
    st.sidebar.error("올바른 Sitemap URL을 입력하세요.")
else:
    retriever = load_retriever(urls)

    # =========================
    # Query 기반 RAG 체인
    # =========================
    chain = (
        {
            "question": RunnablePassthrough(),
            "docs": RunnableLambda(lambda q: retriever.get_relevant_documents(q)),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )

    query = st.text_input("질문을 입력하세요")

    if query:
        with st.spinner("답변 생성 중..."):
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
