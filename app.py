import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import datetime
import json
from fpdf import FPDF

PASSWORD = "StrategiaMA2025"
st.set_page_config(page_title="Strategia MA – Pro", layout="wide")
st.title("📊 Strategia MA – Bot Strategiczny Pro")

password = st.text_input("🔒 Wpisz hasło dostępu:", type="password")
if password != PASSWORD:
    st.stop()

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

def save_chat_history(history):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"history/chat_{now}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return filename

def export_chat_to_pdf(chat):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for entry in chat:
        pdf.multi_cell(0, 10, f"Ty: {entry['question']}")
        pdf.multi_cell(0, 10, f"Strategia MA: {entry['answer']}")
        pdf.ln()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = f"history/chat_{now}.pdf"
    pdf.output(pdf_path)
    return pdf_path

def list_saved_chats():
    return [f for f in os.listdir("history") if f.endswith(".json")]

# 🔽 Wgrywanie plików
uploaded_files = st.file_uploader("📎 Wgraj dokumenty (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("✅ Pliki zostały zapisane!")

# 📂 Lista dokumentów
st.markdown("## 📂 Wgrane dokumenty")
doc_files = os.listdir("docs")
if not doc_files:
    st.info("Brak dokumentów. Wgraj coś powyżej.")
else:
    for file in doc_files:
        st.markdown(f"✅ {file}")

# 📚 Przetwarzanie dokumentów
with st.spinner("📚 Przetwarzanie dokumentów..."):
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# 💬 Czat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("💬 Rozmowa z botem")
user_input = st.text_input("Zadaj pytanie:")
if user_input:
    with st.spinner("🤖 Bot odpowiada..."):
        history_as_pairs = [
            (entry["question"], entry["answer"]) for entry in st.session_state.chat_history
        ]
        result = chain.run({
            "question": user_input,
            "chat_history": history_as_pairs
        })
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": result
        })
        st.markdown(f"**Ty:** {user_input}")
        st.markdown(f"**Bot:** {result}")

# 📄 Eksport i archiwum
if st.session_state.chat_history:
    st.markdown("### 💾 Zapisz rozmowę:")
    if st.button("📤 Eksportuj do PDF"):
        pdf_path = export_chat_to_pdf(st.session_state.chat_history)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Pobierz PDF", f, file_name=os.path.basename(pdf_path))

    if st.button("🗃️ Zapisz rozmowę do archiwum"):
        save_chat_history(st.session_state.chat_history)
        st.success("✅ Rozmowa zapisana do archiwum!")

# 📂 Archiwum rozmów
st.markdown("## 📂 Archiwum rozmów")
history_files = list_saved_chats()
for file in sorted(history_files, reverse=True):
    if st.button(f"📖 {file}"):
        with open(f"history/{file}", "r", encoding="utf-8") as f:
            past_chat = json.load(f)
            for entry in past_chat:
                st.markdown(f"**Ty:** {entry['question']}")
                st.markdown(f"**Bot:** {entry['answer']}")
