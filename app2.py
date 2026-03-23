import streamlit as st
import os
import tempfile
import uuid
from dotenv import load_dotenv

# --- AAPKE SPECIFIC IMPORTS (UNTOUCHED) ---
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.document_loaders import UnstructuredFileLoader, WebBaseLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. API KEY SETUP ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# --- UI SETUP ---
st.set_page_config(page_title="Universal Doc-Intel Pro", layout="wide")

# --- 2. SESSION STATE (Chat History & Sidebar Logic) ---
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {} # {id: {"history": [], "title": ""}}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.all_sessions[st.session_state.current_session_id] = {"history": [], "title": "New Chat"}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- 3. SIDEBAR (Proper History Show Karega) ---
with st.sidebar:
    st.title("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.all_sessions[st.session_state.current_session_id] = {"history": [], "title": "New Chat"}
        st.rerun()
    
    st.divider()
    # Purani chats ki list dikhayega
    for sid, data in st.session_state.all_sessions.items():
        if st.button(data["title"], key=sid, use_container_width=True):
            st.session_state.current_session_id = sid
            st.rerun()

    st.divider()
    st.title("🚀 Data Source")
    source_type = st.radio("Select Source:", ["Local File", "Web URL"])
    
    input_source = None
    if source_type == "Local File":
        input_source = st.file_uploader("Upload File", type=['pdf','png','jpg','csv','docx'])
    else:
        input_source = st.text_input("Enter URL")
    
    process_btn = st.button("⚡ Build RAG Engine")

# --- 4. LOADER FUNCTION (Badi books ke liye optimize kiya) ---
def load_and_split(source, is_url=False):
    if is_url:
        loader = WebBaseLoader(source)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source.name)[-1]) as tmp:
            tmp.write(source.getvalue())
            tmp_path = tmp.name
        
        ext = os.path.splitext(tmp_path)[-1].lower()
        # 200MB+ files ke liye PyPDFLoader automatic use hoga
        if ext == '.pdf': loader = PyPDFLoader(tmp_path)
        elif ext == '.csv': loader = CSVLoader(tmp_path)
        else: loader = UnstructuredFileLoader(tmp_path, strategy="hi_res")
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    return splitter.split_documents(docs)

# LLM (Fast Model)
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

# --- 5. PROCESS BUTTON LOGIC ---
if process_btn and input_source:
    try:
        with st.spinner("Processing... Isme thoda time lag sakta hai."):
            chunks = load_and_split(input_source, is_url=(source_type=="Web URL"))
            valid_chunks = [doc for doc in chunks if doc.page_content.strip()]
            
            embeddings = MistralAIEmbeddings(model="mistral-embed")
            vector_db = Chroma.from_documents(valid_chunks, embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Badi book ke liye k badhaya
            
            # RAG Setup
            context_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Reformulate the question based on chat history. Standalone banayein."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a respectful assistant. Answer using context only. Match user's language (Hinglish/English).\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            st.success("✅ Ready! AI ne document padh liya hai.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- 6. CHAT UI ---
curr_session = st.session_state.all_sessions[st.session_state.current_session_id]
st.title(f"💬 {curr_session['title']}")

# Display History of current session
for msg in curr_session["history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input logic
if user_query := st.chat_input("Type your message..."):
    # Title update karein
    if not curr_session["history"]:
        curr_session["title"] = user_query[:25] + "..."

    st.chat_message("user").write(user_query)
    
    with st.spinner("Thinking..."):
        if st.session_state.rag_chain:
            # RAG Mode with Metadata/Reference
            res = st.session_state.rag_chain.invoke({"input": user_query, "chat_history": curr_session["history"]})
            answer = res["answer"]
            source_docs = res.get("context", [])
        else:
            # General Chat Mode
            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a respectful assistant. Match user language."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            chain = gen_prompt | llm
            res = chain.invoke({"input": user_query, "chat_history": curr_session["history"]})
            answer = res.content
            source_docs = []

        with st.chat_message("assistant"):
            st.write(answer)
            # Metadata Reference Show Karein (Taki lage hallucinate nahi kar raha)
            if source_docs:
                sources = set([doc.metadata.get("source", "Reference") for doc in source_docs])
                st.caption(f"📚 References used: {', '.join(sources)}")

        # Update History
        curr_session["history"].extend([HumanMessage(content=user_query), AIMessage(content=answer)])
