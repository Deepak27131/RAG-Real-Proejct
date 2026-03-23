import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Aapke bataye gaye specific imports
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.document_loaders import UnstructuredFileLoader, WebBaseLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. API KEY SETUP ---
# Yeh line aapki .env file se API key uthayegi
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
else:
    st.error("❌ MISTRAL_API_KEY not found in .env file!")

# --- UI SETUP ---
st.set_page_config(page_title="Universal Doc-Intel Pro", layout="wide")

# Session State for History & Database
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- SIDEBAR (ONLY FOR UPLOADS) ---
with st.sidebar:
    st.title("🚀 Data Source")
    st.info("Mistral API is connected via your system key.")
    
    st.divider()
    source_type = st.radio("Select Source:", ["Local File", "Web URL"])
    
    input_source = None
    if source_type == "Local File":
        input_source = st.file_uploader("Upload File", type=['pdf','png','jpg','csv','docx'])
    else:
        input_source = st.text_input("Enter URL")
    
    process_btn = st.button("⚡ Build RAG Engine")

# --- LOADER FUNCTION ---
def load_and_split(source, is_url=False):
    if is_url:
        loader = WebBaseLoader(source)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source.name)[-1]) as tmp:
            tmp.write(source.getvalue())
            tmp_path = tmp.name
        
        ext = os.path.splitext(tmp_path)[-1].lower()
        if ext == '.csv': loader = CSVLoader(tmp_path)
        elif ext == '.pdf': loader = PyPDFLoader(tmp_path)
        else: loader = UnstructuredFileLoader(tmp_path, strategy="hi_res")
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- LLM INITIALIZATION (Used for both RAG and General Chat) ---
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.2)

# --- PROCESS BUTTON LOGIC ---
if process_btn and input_source:
    try:
        with st.spinner("Processing document..."):
            chunks = load_and_split(input_source, is_url=(source_type=="Web URL"))
            
            # Metadata fix: Ensure no empty content
            valid_chunks = [doc for doc in chunks if doc.page_content.strip()]
            
            embeddings = MistralAIEmbeddings(model="mistral-embed")
            vector_db = Chroma.from_documents(valid_chunks, embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            
            # RAG Setup
            context_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Reformulate the question based on chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question using context:\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            st.success("✅ Document Indexed! AI will now answer from this doc.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- CHAT UI ---
st.title("💬 Intelligent Assistant")
if not st.session_state.rag_chain:
    st.caption("🌐 Currently in **General Chat** mode (No document uploaded)")
else:
    st.caption("📄 Currently in **Document RAG** mode")

# Display History
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input logic
if user_query := st.chat_input("Type your message..."):
    st.chat_message("user").write(user_query)
    
    # 2. DECISION LOGIC: RAG vs General Chat
    if st.session_state.rag_chain:
        # RAG Mode
        res = st.session_state.rag_chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        })
        answer = res["answer"]
    else:
        # General Chat Mode (Bina PDF/URL ke chalne wala)
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        chain = general_prompt | llm
        res = chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        })
        answer = res.content

    # Update UI and History
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query), 
        AIMessage(content=answer)
    ])
