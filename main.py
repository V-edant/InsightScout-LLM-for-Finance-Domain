import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import os
import time # Import the time module

# Set your Gemini API key here
API_KEY = "Replace with your actual API key" 
genai.configure(api_key=API_KEY)

# Initialize model
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI
st.title("🌐 Insight Scout: LLM for Finance Domain")

# Input for multiple URLs
url1 = st.text_input("Enter URL 1:")
url2 = st.text_input("Enter URL 2:")
url3 = st.text_input("Enter URL 3:")

# Process button
if st.button("Process URLs"):
    # Collect non-empty URLs
    urls = [url for url in [url1, url2, url3] if url]

    if not urls:
        st.warning("Please enter at least one URL")
    else:
        try:
            with st.spinner(f"Loading content from {len(urls)} URLs..."):
                # Store URL to document mapping for source attribution
                url_to_docs = {}
                all_docs = []

                # Process each URL individually
                for url in urls:
                    loader = UnstructuredURLLoader(urls=[url])
                    docs = loader.load()

                    # Store the mapping from URL to documents
                    url_to_docs[url] = docs
                    all_docs.extend(docs)

                # Create a text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )

                # Split documents and track their source URLs
                all_chunks = []
                chunk_sources = {}  # Map each chunk to its source URL

                for url, docs in url_to_docs.items():
                    for doc in docs:
                        chunks = text_splitter.split_text(doc.page_content)
                        for chunk in chunks:
                            chunk_id = len(all_chunks)
                            all_chunks.append(chunk)
                            chunk_sources[chunk_id] = url

                # Create embeddings and store in FAISS
                with tempfile.TemporaryDirectory() as temp_dir:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=API_KEY
                    )

                    # Create vector store
                    vector_store = FAISS.from_texts(all_chunks, embeddings)

                    # Add a small delay after creating the vector store (after embeddings are generated)
                    time.sleep(10) # Sleep for 2 seconds

                    st.success(f"✅ Content extracted from {len(urls)} URLs. Now ask your question.")

                    # Store in session state to persist between reruns
                    st.session_state['vector_store'] = vector_store
                    st.session_state['chunk_sources'] = chunk_sources
                    st.session_state['all_chunks'] = all_chunks
                    st.session_state['processed'] = True

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("If you're seeing an error related to UnstructuredURLLoader, you may need to install additional dependencies.")
            st.code("pip install unstructured nltk")

# Question handling - only show if URLs have been processed
if st.session_state.get('processed', False):
    question = st.text_input("What do you want to know about this content?")

    if question:
        with st.spinner("Finding relevant information and generating answer..."):
            # Retrieve from session state
            vector_store = st.session_state['vector_store']
            chunk_sources = st.session_state['chunk_sources']
            all_chunks = st.session_state['all_chunks']

            # Search for relevant chunks
            search_results = vector_store.similarity_search_with_score(question, k=3)

            # Extract relevant chunks and their sources
            relevant_chunks = []
            source_urls = []

            for doc, score in search_results:
                # Find the original chunk
                chunk_text = doc.page_content
                # We need to find the index of the chunk in all_chunks to get its source
                try:
                    chunk_idx = all_chunks.index(chunk_text)
                    source_url = chunk_sources[chunk_idx]
                except ValueError:
                    # This can happen if the doc.page_content isn't directly in all_chunks
                    # (e.g., due to minor internal processing differences or if doc includes metadata)
                    # For robust source attribution, you might need to store metadata with chunks.
                    source_url = "Unknown Source" # Fallback
                
                relevant_chunks.append(chunk_text)
                source_urls.append(source_url)

            # Combine chunks for context
            context = "\n\n".join(relevant_chunks)

            # Generate answer with Gemini
            prompt = f"""Use the following page content to answer the question accurately.
            If the information is not present in the content, say so honestly.

            Page Content:
            {context}

            Question:
            {question}
            """
            
            # Add a crucial delay before making the content generation API call
            time.sleep(10) # Sleep for 5 seconds (adjust as needed)
            response = model.generate_content(prompt)

            # Display the answer
            st.subheader("Answer:")
            st.markdown(response.text)

            # Show source links
            st.subheader("Sources:")
            unique_sources = list(set(source_urls))
            for i, source in enumerate(unique_sources):
                st.markdown(f"- [{source}]({source})")

# Reset button to clear the session state
if st.session_state.get('processed', False):
    if st.button("Process New URLs"):
        for key in ['vector_store', 'chunk_sources', 'all_chunks', 'processed']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
