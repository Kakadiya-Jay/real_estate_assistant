import os

# 1. SILENCE TENSORFLOW WARNINGS
# This must be done BEFORE importing other libraries
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3 = Suppress all logs except fatal errors

import streamlit as st

# Make sure your rag.py file is in the same folder
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

# Sidebar
st.sidebar.header("Data Sources")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")

# Main Page Layout
# We create a specific container just for status messages
status_container = st.empty()

# LOGIC: Processing URLs
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]

    if not urls:
        status_container.error("You must provide at least one valid URL.")
    else:
        status_container.info("Processing started...")
        try:
            # We assume process_urls is a generator or returns text
            for status in process_urls(urls):
                status_container.text(status)

            status_container.success("Processing Complete! You can now ask questions.")
        except Exception as e:
            status_container.error(f"Error during processing: {e}")

# LOGIC: Question Answering
# We use a standard text_input, distinct from the status_container
query = st.text_input("Question")

if query:
    # Use a spinner to show activity while generating
    with st.spinner("Generating answer..."):
        try:
            answer, sources = generate_answer(query)

            st.header("Answer:")
            st.markdown(answer)  # Markdown looks better for AI answers

            if sources:
                st.subheader("Sources:")
                # Based on our previous fix, 'sources' is a string.
                # If it's a comma-separated string, we can just display it.
                st.write(sources)

        except Exception as e:
            # This catches the "Components not initialized" error
            st.error(
                f"Could not generate answer. (Did you click 'Process URLs' first?)"
            )
            st.caption(f"Technical details: {e}")
