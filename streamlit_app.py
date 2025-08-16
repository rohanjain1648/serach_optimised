import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle API key for both local and Streamlit Cloud
def get_nvidia_api_key():
    """Get NVIDIA API key from Streamlit secrets or environment variables."""
    
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets.get("NVIDIA_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass
    
    # Try environment variable (for local development)
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        return api_key
    
    # If no key found, show error
    st.error("""
    **NVIDIA API Key Not Found!**
    
    To use this app, you need to set up your NVIDIA API key:
    
    **For Streamlit Cloud:**
    1. Go to your app settings
    2. Click on 'Secrets'
    3. Add: `NVIDIA_API_KEY = "your-key-here"`
    
    **For Local Development:**
    1. Create a `.env` file in your project root
    2. Add: `NVIDIA_API_KEY=your-key-here`
    
    Get your API key from: https://build.nvidia.com/explore/discover
    """)
    return None

def vector_embedding(api_key):
    """Create vector embeddings for documents."""
    try:
        if "vectors" not in st.session_state:
            with st.spinner("Creating embeddings... This may take a moment."):
                # Set API key
                os.environ['NVIDIA_API_KEY'] = api_key
                
                # Initialize NVIDIA embeddings
                st.session_state.embeddings = NVIDIAEmbeddings(
                    model="nvidia/nv-embedqa-e5-v5"
                )
                
                # Load PDF documents
                st.session_state.loader = PyPDFDirectoryLoader("./us_census")
                st.session_state.docs = st.session_state.loader.load()
                
                if not st.session_state.docs:
                    st.error("No documents found in the us_census directory.")
                    return False
                
                # Split documents into chunks
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=50
                )
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                    st.session_state.docs[:40]
                )
                
                # Create vector store
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents,
                    st.session_state.embeddings
                )
                return True
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return False

# Streamlit UI
st.title("NVIDIA NIM Demo with RAG")
st.markdown("### Retrieval-Augmented Generation with NVIDIA AI")

# Get API key
api_key = get_nvidia_api_key()
if not api_key:
    st.stop()

# Initialize LLM
try:
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.7,
        max_tokens=1024
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
)

# User input
prompt1 = st.text_input("Enter your question about the documents:")

# Document embedding button
if st.button("Create Document Embeddings"):
    if vector_embedding(api_key):
        st.success("✅ FAISS Vector Store is ready using NVIDIA Embeddings!")
        st.info(f"Loaded {len(st.session_state.final_documents)} document chunks")

# Query processing
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first by clicking the button above.")
    else:
        try:
            with st.spinner("Processing your question..."):
                # Create chains
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Measure response time
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                # Display results
                st.success(f"Response generated in {response_time:.2f} seconds")
                st.markdown("### Answer:")
                st.write(response['answer'])
                
                # Show relevant document chunks
                with st.expander("View relevant document excerpts"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Excerpt {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Sidebar information
with st.sidebar:
    st.markdown("### About")
    st.info(
        "This app uses NVIDIA's AI endpoints for:\n"
        "- Document embeddings (nv-embedqa-e5-v5)\n"
        "- Question answering (Llama 3.3 70B)\n"
        "- Retrieval-augmented generation"
    )
    
    st.markdown("### Setup Requirements")
    st.warning(
        "Ensure you have:\n"
        "- Valid NVIDIA API key configured\n"
        "- PDF documents in ./us_census directory\n"
        "- Internet connection for API calls"
    )
