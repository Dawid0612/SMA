
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import datetime

# Hasło dostępu
PASSWORD = "StrategiaMA2025"

st.set_page_config(page_title="Strategia MA - Bot", layout="wide")
st.title("🤖 Strategia MA - Twój Bot Strategiczny")

password = st.text_input("🔒 Wpisz hasło, aby uzyskać dostęp:", type="password")
if password != PASSWORD:
    st.warning("Niepoprawne hasło.")
    st.stop()

# Ładowanie klucza z sekretów Streamlit Cloud
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)

os.makedirs("docs", exist_ok=True)
os.makedirs("history", exist_ok=True)

def load_documents():
    documents = []
    for filename in os.listdir("docs"):
        filepath = os.path.join("docs", filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents

uploaded_files = st.file_uploader("📎 Wgraj dokumenty (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("✅ Pliki zostały zapisane!")

with st.spinner("🔄 Przetwarzanie dokumentów..."):
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Zadaj pytanie:")
if user_input:
    with st.spinner("🤖 Bot odpowiada..."):
        result = chain.run({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_input, result))
        st.markdown(f"**Ty:** {user_input}")
        st.markdown(f"**Bot Strategia MA:** {result}")

if st.session_state.chat_history:
    save_button = st.button("💾 Zapisz historię rozmowy")
    if save_button:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join("history", f"chat_{now}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            for q, a in st.session_state.chat_history:
                f.write(f"Ty: {q}\nStrategia MA: {a}\n\n")
        st.success(f"✅ Historia zapisana: {filename}")
