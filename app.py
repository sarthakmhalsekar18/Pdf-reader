import os
import tempfile
import asyncio
import streamlit as st
# try:
# from dotenv import load_dotenv
# load_dotenv()
# st.write("🔑 DEBUG: Loaded API key =", os.getenv("GOOGLE_API_KEY"))
# except ImportError:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv==1.0.0"])
#     from dotenv import load_dotenv

# Configure page before other imports
st.set_page_config(
    page_title="Gemini PDF Q&A Assistant",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Import with detailed error handling
try:
    from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings, 
        ChatGoogleGenerativeAI,
        HarmCategory,
        HarmBlockThreshold
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    # st.write("✅ All required packages imported successfully")
except ImportError as e:
    st.error(f"❌ Missing required packages: {str(e)}")
    st.info("Please install requirements: pip install langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv")
    st.stop()

# Check API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("""
    ❌ Google API key not found. Please:
    1. Create a .env file in your project folder
    2. Add: GOOGLE_API_KEY=your_actual_key_here
    3. Restart the application
    """)
    st.stop()
# else:
#     st.write("✅ Google API key loaded from environment variables")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def validate_pdf(file):
    """Check if uploaded file is a valid PDF"""
    try:
        file.seek(0)
        header = file.read(4)
        is_pdf = header == b'%PDF'
        st.write(f"🔍 PDF Validation: {'✅ Valid PDF' if is_pdf else '❌ Not a PDF'}")
        return is_pdf
    except Exception as e:
        st.write(f"❌ PDF Validation Error: {str(e)}")
        return False

async def process_pdf_async(uploaded_file):
    """Async PDF processing with detailed logging"""
    st.write("🔄 Starting PDF processing...")
    tmp_file_path = None
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        st.write(f"📄 Created temp file at: {tmp_file_path}")

        # Load PDF
        st.write("📖 Loading PDF pages...")
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        st.write(f"📚 Loaded {len(pages)} pages")

        # Split text
        st.write("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        st.write(f"🧩 Created {len(chunks)} text chunks")

        # Create embeddings
        st.write("🔢 Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create vector store
        st.write("🗄️ Creating vector store...")
        vector_store = await FAISS.afrom_documents(chunks, embeddings)
        st.write("✅ Vector store created successfully")
        
        return vector_store
    
    except Exception as e:
        st.error(f"❌ PDF Processing Error: {str(e)}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                st.write("🧹 Cleaned up temp file")
            except Exception as e:
                st.write(f"⚠️ Temp file cleanup error: {str(e)}")

def initialize_qa(vector_store):
    """Initialize QA system with detailed validation"""
    st.write("🧠 Initializing QA system...")
    
    try:
        # Validate vector store
        if not vector_store:
            raise ValueError("No valid vector store provided")
        
        st.write("🔍 Creating retriever...")
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        )
        st.write("✅ Retriever created")

        st.write("🤖 Configuring Gemini model...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=2000,
            google_api_key=GOOGLE_API_KEY,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        st.write("✅ Gemini model configured")

        st.write("📝 Setting up prompt template...")
        prompt_template = """Answer based on this context:
        
        {context}
        
        Question: {question}
        
        Guidelines:
        - Be accurate and cite sources like [Page X]
        - Say "Not in document" if unsure
        - Use bullet points when helpful
        
        Answer:"""
        
        PROMPT = PromptTemplate.from_template(prompt_template)
        st.write("✅ Prompt template ready")

        st.write("⛓️ Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        st.write("✅ QA system initialized successfully")
        
        return qa_chain
    
    except Exception as e:
        st.error(f"❌ QA System Error: {str(e)}")
        return None

# UI Components
st.title("📄 Debugging PDF Analyzer")
st.markdown("""
<style>
.stChatInput {position: fixed; bottom: 2rem;}
.stChatMessage {padding: 1rem;}
</style>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    # st.header("Debug Controls")
    # debug_mode = st.checkbox("Enable Debug Output", True)
    uploaded_file = st.file_uploader("Choose PDF File", type="pdf")
    
    if uploaded_file:
        st.write("📂 File uploaded successfully")
        if not validate_pdf(uploaded_file):
            st.error("Invalid PDF file")
        elif not st.session_state.pdf_processed:
            with st.spinner("Processing document..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    vector_store = loop.run_until_complete(process_pdf_async(uploaded_file))
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = initialize_qa(vector_store)
                        st.session_state.pdf_processed = True
                    else:
                        st.error("Failed to process document")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")

    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# Chat Interface
if st.session_state.qa_chain:
    st.write("💬 Chat interface ready")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    st.write("⚙️ Processing query...")
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    
                    if not response or "result" not in response:
                        raise ValueError("Invalid response from QA chain")
                    
                    answer = response["result"]
                    st.write("✅ Response generated")
                    
                    sources = []
                    if "source_documents" in response:
                        for doc in response["source_documents"]:
                            page_num = doc.metadata.get('page', 0) + 1
                            content = doc.page_content[:200].replace('\n', ' ').strip()
                            sources.append(f"📖 Page {page_num}: {content}...")
                    
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
                    st.error(f"❌ Response Error: {str(e)}")
else:
    st.info("👈 Upload a PDF document to begin analysis")
    st.markdown("""
    ### How to Use:
    1. Upload a PDF document
    2. View debug output in the console
    3. Ask questions about the content
    4. See detailed processing steps
    
    🔍 Enable debug mode for detailed logs
    """)

# if debug_mode and st.session_state.vector_store:
#     with st.expander("Debug Information"):
#         st.write("### Vector Store Info")
#         st.write(f"Index size: {len(st.session_state.vector_store.index_to_docstore_id)} documents")
#         st.write("### Session State")
#         st.json({k: str(v) for k, v in st.session_state.items()})