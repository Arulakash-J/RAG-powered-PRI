import streamlit as st
import os
from tools import process_pdf_document, retrieve_relevant_chunks, highlight_matching_chunks, get_pdf_download_link

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'highlighted_chunks' not in st.session_state:
        st.session_state.highlighted_chunks = []
    if 'document_source' not in st.session_state:
        st.session_state.document_source = None

def create_ui(app_title, file_size_limit_mb):
    """Create the main UI components"""
    st.set_page_config(page_title=app_title, page_icon="📄", layout="wide")
    st.title(f"{app_title} 📄")
    st.markdown("Upload a PDF and ask questions about its content!")

def load_example_resume(tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap):
    """Load the example resume for testing"""
    example_resume_path = "./Arul Akash(AI_ML Developer).pdf"  # Path to your example resume
    
    if not os.path.exists(example_resume_path):
        st.error(f"Example resume not found at {example_resume_path}")
        return False
    
    try:
        with open(example_resume_path, "rb") as file:
            with st.spinner("Processing Resume..."):
                success, chunk_count = process_pdf_document(
                    file, tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap
                )
                
                if success:
                    st.session_state.document_loaded = True
                    st.session_state.document_name = "Arul_Akash_Resume.pdf"
                    st.session_state.success_message = "✅ Ready to Rock!"
                    st.session_state.chunk_count = chunk_count
                    return True
                else:
                    st.error("❌ Failed to process example resume")
                    return False
    except Exception as e:
        st.error(f"Error loading example resume: {str(e)}")
        return False

def create_sidebar(tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap, file_size_limit_mb):
    """Create the sidebar upload functionality"""
    with st.sidebar:
        st.header("📁 Document Upload")

        # Add custom styling for the button
        st.markdown("""
        <style>
        /* Custom styling for the example resume button */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(to right, #4776E6, #8E54E9);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
            background: linear-gradient(to right, #5a85f0, #9d65f0);
        }
        /* Adding style for disabled buttons */
        div[data-testid="stButton"] button[disabled] {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Store the current document source (resume or uploaded) in session state
        if 'document_source' not in st.session_state:
            st.session_state.document_source = None
            
        # Check if resume is already loaded
        resume_already_loaded = (st.session_state.document_source == 'resume' and 
                               st.session_state.document_loaded and 
                               st.session_state.document_name == "Arul_Akash_Resume.pdf")
        
        # Only show the resume button if resume is not already loaded
        if not resume_already_loaded:
            # Attractive colored button (disabled if file is uploaded)
            resume_button_disabled = st.session_state.document_source == 'upload'
            if st.button("📄 Try Arul Akash Resume", 
                        help="Load example resume for testing",
                        use_container_width=True,
                        type="primary",
                        disabled=resume_button_disabled):
                success = load_example_resume(tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap)
                if success:
                    st.session_state.document_source = 'resume'
                    # Success message will be shown outside of this if block
        else:
            # Show status message for already loaded resume
            st.info("Resume is currently loaded. To upload a different document, refresh the page.")
        
        # Show persistent success message if resume is loaded
        if resume_already_loaded:
            st.success("✅ Ready to Rock!")

        # Handle file uploads
        # Only show file uploader if resume is not loaded
        if st.session_state.document_source != 'resume':
            uploaded_file = st.file_uploader(
                "Upload PDF", 
                type=["pdf"],
                help=f"Maximum file size: {file_size_limit_mb}MB"
            )
            
            # Display the file size limit explicitly
            st.caption(f"Limit {file_size_limit_mb}MB per file • PDF")

            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024 * 1024) 
                if file_size_mb > file_size_limit_mb:
                    st.error(f"File size ({file_size_mb:.1f} MB) exceeds the {file_size_limit_mb} MB limit. Please upload a smaller file.")
                    st.session_state.document_loaded = False
                    st.session_state.document_name = None
                    st.session_state.document_source = None
                elif st.session_state.document_name != uploaded_file.name:
                    st.session_state.document_loaded = False
                    st.session_state.document_name = uploaded_file.name
                
                if not st.session_state.document_loaded:
                    with st.spinner("Processing document..."):
                        success, chunk_count = process_pdf_document(
                            uploaded_file, tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap
                        )
                        if success:
                            st.session_state.document_loaded = True
                            st.session_state.document_source = 'upload'
                            # Create a container for the success message and store it in session state
                            st.success(f"✅ Document '{uploaded_file.name}' processed into {chunk_count} chunks!")

                            with open(uploaded_file.name, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            st.markdown(get_pdf_download_link(uploaded_file.name), unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to process document")
                else:
                    # Show success message consistently for uploaded file
                    st.success(f"✅ Document '{uploaded_file.name}' is loaded and ready for questions!")

                    if os.path.exists(uploaded_file.name):
                        st.markdown(get_pdf_download_link(uploaded_file.name), unsafe_allow_html=True)
                
                return uploaded_file
        elif st.session_state.document_source == 'upload':
            # If we have switched to showing the resume, but previously had a file upload active
            st.info("A file was previously uploaded. To switch back to it, refresh the page.")
        
        return None


def load_example_resume(tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap):
    """Load the example resume for testing"""
    example_resume_path = "./Arul Akash(AI_ML Developer).pdf"  # Path to your example resume
    
    if not os.path.exists(example_resume_path):
        st.error(f"Example resume not found at {example_resume_path}")
        return False
    
    try:
        with open(example_resume_path, "rb") as file:
            with st.spinner("Processing My Resume..."):
                success, chunk_count = process_pdf_document(
                    file, tokenizer, embedding_model, pinecone_index, chunk_size, chunk_overlap
                )
                
                if success:
                    st.session_state.document_loaded = True
                    st.session_state.document_name = "Arul_Akash_Resume.pdf"
                    # Display success message right after loading
                    st.success("✅ Ready to Rock!")
                    return True
                else:
                    st.error("❌ Failed to process example resume")
                    return False
    except Exception as e:
        st.error(f"Error loading example resume: {str(e)}")
        return False

def create_chat_interface(groq_client, tokenizer, embedding_model, pinecone_index, llm_model, generate_response_func):
    """Create the chat interface for Q&A"""
    st.header("💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "highlighted_chunks" in message:
                with st.expander("View Source Chunks"):
                    for i, chunk in enumerate(message["highlighted_chunks"]):
                        st.markdown(f"**Source Chunk {i+1}** (from {chunk['source']}):")
                        st.text(chunk["text"])
    
    # User input
    if user_query := st.chat_input("Ask a question about the document..."):
        if not st.session_state.document_loaded:
            with st.chat_message("assistant"):
                st.markdown("⚠️ Please upload a document first before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chunks = retrieve_relevant_chunks(
                        user_query, pinecone_index, tokenizer, embedding_model
                    )

                    response = generate_response_func(user_query, chunks, groq_client, llm_model)

                    st.markdown(response)

                    highlighted_chunks = highlight_matching_chunks(response, chunks)
                    
                    if highlighted_chunks:
                        with st.expander("View Source Chunks"):
                            for i, chunk in enumerate(highlighted_chunks):
                                st.markdown(f"**Source Chunk {i+1}** (from {chunk['source']}):")
                                st.text(chunk["text"])
            
            # Add assistant message with highlighted chunks to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "highlighted_chunks": highlighted_chunks
            })