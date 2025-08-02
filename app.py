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
st.set_page_config(
    page_title="Gemini PDF Q&A Assistant",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ Google API key not found. Please add it to your environment variables.")
    st.stop()

# PDF Processing Function with error handling
def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
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
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    finally:
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)

# Initialize QA System with enhanced prompt
def initialize_qa(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=2000,
        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
    )
    
    prompt_template = """
    You are an expert AI assistant analyzing PDF documents. Follow these guidelines:
    
    Context: {context}
    
    Question: {question}
    
    Response Requirements:
    1. Answer strictly based on the provided context
    2. Be concise yet comprehensive
    3. Format with bullet points when listing items
    4. Always cite page numbers like [Page X]
    5. If unsure, say "The document doesn't specify"
    
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
st.title("📄 Gemini-Powered PDF Analyzer")
st.markdown("""
    <style>
    .stChatInput {position: fixed; bottom: 2rem;}
    .stChatMessage {padding: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for additional controls
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader"
    )
    
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            vector_store = process_pdf(uploaded_file)
            if vector_store:
                st.session_state.qa_chain = initialize_qa(vector_store)
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
            else:
                st.error("Failed to process PDF")

# Chat Interface
if st.session_state.qa_chain:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = st.session_state.qa_chain({"query": prompt})
                    answer = response["result"]
                    
                    # Format sources
                    sources = []
                    for doc in response["source_documents"]:
                        page_num = doc.metadata.get('page', 0) + 1
                        sources.append(f"📄 Page {page_num}: {doc.page_content[:200]}...")
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("🔍 Source References"):
                            for source in sources:
                                st.markdown(f"- {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

else:
    st.info("👈 Please upload a PDF file to begin")
    st.markdown("""
    ### How to use:
    1. Upload a PDF document
    2. Ask questions about its content
    3. Get accurate answers with source references
    """)