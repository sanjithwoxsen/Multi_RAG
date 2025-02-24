import time

import streamlit as st
import Rag
import os

st.title("PDF AI")
st.header("Chat With Multiple PDF")
# Sidebar for file upload and model selection
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Choose PDF Files", accept_multiple_files=True, type="pdf")

    if st.button("Submit and Process"):
        with st.spinner("Processing..."):
            raw_text = Rag.pdf_to_text(pdf_docs)
            text_chunks = Rag.text_to_chunks(raw_text)
            Rag.vectorstore_universal(text_chunks)
            st.success("Vectorization Complete")

    option = st.selectbox("Select Model :", ("Gemini-1.5-Pro (API)","Gemma-2B"))
    st.write("Clear Knowledge Base :")
    if st.button("Clear"):
        info_placeholder = st.empty()

        a = Rag.Clear()
        if a == True:
            info_placeholder.success("Cleared KnowledgeBase")

        elif a==-1:
            info_placeholder.error("permission errors")
        else:
            info_placeholder.warning("KnowledgeBase Already Cleared")
            time.sleep(3)
            info_placeholder.empty()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("Ask a Question")

if prompt:
    # Store user question
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for typing indicator

        # Model selection and response generation
        if option == "Flan-T5-base(Fine-Tuned)":
            Model = Rag.Flan_T5_base(user_question=prompt)
        if option == "Mistral":
            Model = Rag.Mistral(user_question=prompt)
        if option == "Gemma":
            Model = Rag.Gemma(user_question=prompt)
        if option == "Gemini-1.5-Pro (API)":
            Model = Rag.GeminiPro(user_question=prompt)
        response = Model.generate()

        # Handle response format based on model
        if option == "Gemini-1.5-Pro (API)":
            full_response = response["output_text"]
        else:
            full_response = response

        # Replace placeholder with actual response
        message_placeholder.markdown(f"{option} : "+full_response)

        # Store bot response
        st.session_state.messages.append({"role": "assistant", "content": f"{option}:"+full_response})
