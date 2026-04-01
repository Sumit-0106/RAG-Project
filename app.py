import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# --- SAME BACKEND (no changes) ---
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# --- UI PART ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot (Mistral AI)")
st.write("Upload a PDF and ask questions from it")

# 📄 File Upload
uploaded_file = st.file_uploader("📄 Upload your PDF / Book", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    vectorstore.add_documents(docs)

    st.success("✅ PDF uploaded and processed!")

# 💬 Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 Input Box
query = st.text_input("💬 Ask a question:")

if st.button("Ask") and query:
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("AI", response.content))

# 💬 Display Chat
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")