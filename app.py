import os
import tempfile
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure page
st.set_page_config(page_title="Gemini 1.5 Flash Q&A", layout="wide")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load environment variables (for local testing)
from dotenv import load_dotenv
load_dotenv()

# PDF Processing Function
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        vector_store = FAISS.from_documents(
            chunks, 
            embeddings,
            distance_strategy="COSINE"
        )
        
        return vector_store
    
    finally:
        os.unlink(tmp_file_path)

# Initialize QA System
def initialize_qa(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=2000
    )
    
    prompt_template = """
    You are an expert at answering questions based on provided documents.
    Use this context to provide a detailed, accurate response:
    
    Context: {context}
    
    Question: {question}
    
    Guidelines:
    1. Be precise and factual
    2. Include relevant details
    3. Structure your answer clearly
    4. Mention page numbers if applicable
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# UI Components
st.title("📄 Gemini 1.5 Flash PDF Q&A")
st.caption("Upload a PDF and ask questions about its content")

# File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.qa_chain:
    with st.spinner("Processing PDF with Gemini 1.5 Flash..."):
        vector_store = process_pdf(uploaded_file)
        st.session_state.qa_chain = initialize_qa(vector_store)
    st.success("PDF processed successfully!")

# Chat Interface
if st.session_state.qa_chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain({"query": prompt})
                st.markdown(response["result"])
                
                with st.expander("View sources"):
                    for doc in response["source_documents"]:
                        st.caption(f"Page {doc.metadata.get('page', 0)+1}")
                        st.text(doc.page_content[:300] + "...")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["result"]
            })