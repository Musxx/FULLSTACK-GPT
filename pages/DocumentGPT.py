import os
import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import chroma
from langchain.embeddings import OpenAIEmbeddings,CacheBackedEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ChatMessageHistory


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="default")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.info("OpenAI API Key를 입력하세요.")
    st.stop()

# # 하는 역할 : DocumentGPT 페이지를 생성하고, 사용자와의 채팅 인터페이스를 제공하여 문서 관련 질문에 답변합니다.
# st.title("Home Page")
# st.write("Welcome to the Home Page of our Streamlit application!")


# # 하는 역할 : DocumentGPT 페이지에서 채팅 인터페이스를 설정하고, 사용자의 질문에 답변을 제공합니다.
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []


# # 하는 역할 : 채팅 메시지를 화면에 표시하고, 세션 상태에 저장하는 함수입니다.
# def send_message(message,role,save=True): 
#     with st.chat_message(role):
#         st.write(message)
#     if save:
#         st.session_state["messages"].append({"role": role, "content": message})

# # 하는 역할 :이전에 저장된 메시지를 세션 상태에서 불러와 화면에 표시합니다.
# for message in st.session_state["messages"]:
#     send_message(message["content"],message["role"],save=False)

# # 하는 역할 :사용자로부터 입력을 받아 채팅 인터페이스를 통해 질문을 처리합니다.
# message = st.chat_input("Ask me anything about DocumentGPT...")   

# # 하는 역할 :사용자가 메시지를 입력하면, 해당 메시지를 세션 상태에 저장하고, 간단한 답변을 생성하여 화면에 표시합니다.
# if message:
#     send_message(message,"user")
#     send_message(f"You said: {message}","assistant")

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "streaming_done" not in st.session_state:
    st.session_state.streaming_done = True

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""




class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.box = None

    def on_llm_start(self, *args, **kwargs):
        self.text = ""
        self.box = st.empty()
        st.session_state.streaming_done = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.box.markdown(self.text)

    def on_llm_end(self, *args, **kwargs):
        st.session_state.streaming_done = True
        st.session_state.last_answer = self.text

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[CustomCallbackHandler()])

# 하는 역할 :대화 요약 버퍼 메모리 인스턴스 생성
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=50, return_messages=True)


st.title("DocumentGPT Page")

st.markdown("""
Welcome to the DocumentGPT page! Here, you can interact with the DocumentGPT model to ask questions about documents.

""")

# 하는 역할 :문서 업로드 위젯 생성
file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

# 하는 역할 : 채팅 메시지를 화면에 표시하고, 세션 상태에 저장하는 함수입니다.
# @st.cache_data(show_spinner="Reading document...")
# def load_and_split(file):
#     file_content = file.read()
#     file_dir = f"./.cache/files/{file.name}"
#     os.makedirs(file_dir, exist_ok=True)
#     file_path = f"{file_dir}/{file.name}"
#     with open(file_path, "wb") as f:
#         f.write(file_content)

#     # 하는 역할 :문서 로더 및 텍스트 분할기 인스턴스 생성
#     loader = UnstructuredFileLoader(file_path)

#     #  하는 역할 :텍스트 분할기 설정
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)

#     # 하는 역할 :문서 로드 및 분할
#     documents = loader.load_and_split(text_splitter=splitter)
#     texts = splitter.split_documents(documents)

#     return texts

# @st.cache_data(show_spinner="Reading document...")
def load_and_split(file_name: str, file_bytes: bytes):
    base_dir = "./.cache/files"
    os.makedirs(base_dir, exist_ok=True)

    file_path = os.path.join(base_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    loader = UnstructuredFileLoader(file_path)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = loader.load()
    texts = splitter.split_documents(documents)
    return texts


# 하는 역할 : 채팅 메시지를 세션 상태에 저장하는 함수입니다.
def save_message(message,role):
    st.session_state["messages"].append({"role": role, "content": message})

# 하는 역할 : 채팅 메시지를 화면에 표시하고, 세션 상태에 저장하는 함수입니다.
def send_message(message,role,save=True): 
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message,role)

# 하는 역할 :대화 기록 불러오는 함수 정의
def load_history(_):
    return memory.load_memory_variables({})["history"]

# 하는 역할 :이전에 저장된 메시지를 세션 상태에서 불러와 화면에 표시합니다.
def chat_history():
    for message in st.session_state["messages"]:
        send_message(message["content"],message["role"],save=False)

# 하는 역할 :임베딩 생성 및 검색기 빌드 함수
# @st.cache_resource(show_spinner="Embedding document...")
def build_retriever(texts,file_name):
    # 하는 역할 :로컬 파일 스토어 인스턴스 생성
    emb_dir = (f"./.cache/embeddings/{file_name}/") 
    os.makedirs(emb_dir, exist_ok=True)
    storage = LocalFileStore(emb_dir)

    # 하는 역할 :임베딩 모델 인스턴스 생성
    embeddings = OpenAIEmbeddings()


    # 하는 역할 :캐시 백드 임베딩 인스턴스 생성
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, storage )  #하는 역할 :로컬 파일 스토어를 사용하여 임베딩 캐싱


    # 하는 역할 :벡터 스토어 생성
    vectorstore = chroma.Chroma.from_documents(texts, cached_embeddings) # 하는 역할 :문서와 캐시 백드 임베딩을 사용하여 벡터 스토어 생성

    retriever = vectorstore.as_retriever()

    return retriever

# 하는 역할 :문서 목록을 단일 텍스트로 변환하는 함수
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 질문에 간단히 답하세요. 모른다면 모른다고 답해주세요.\n\n{context}"),
    ("user", "{question}")
])

# 하는 역할 :문서가 업로드되면, 문서를 로드하고 분할한 후, 임베딩을 생성하고 검색기를 빌드합니다. 이후, 사용자의 질문에 대해 답변을 생성합니다.
if file:
    texts = load_and_split(file.name, file.getvalue())
    retriever = build_retriever(texts,file.name)
    chat_history()
    message = st.chat_input("Ask me anything about the document you uploaded...")   
    if message:
        send_message(message,"user")
        qa_chain = {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm

        with st.chat_message("ai"):
            with st.spinner("Generating response..."):
                qa_chain.invoke(message)

        # ✅ 스트리밍 종료 후 단 한 번만
        if st.session_state.streaming_done:
            # UI용 메시지 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": st.session_state.last_answer
            })

            # ✅ Memory 저장
            memory.save_context(
                {"input": message},
                {"output": st.session_state.last_answer}
            )
else:
    st.session_state["messages"] = []
