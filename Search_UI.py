import streamlit as st
import Rag

st.set_page_config("Chat With Multiple PDF")
st.header("PDF AI")
user_question = st.text_input("Ask a Question")
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Choose PDF Files",accept_multiple_files=True,type="pdf")
    if st.button("Submit and Process"):
        with st.spinner("Processing..."):
            raw_text = Rag.pdf_to_text(pdf_docs)
            text_chunks = Rag.text_to_chunks(raw_text)
            Rag.vectorstore_universal(text_chunks)
            st.success("Vectorization Complete")
    option = st.selectbox("Select Model", ('Flan-T5-base(Fine-Tuned)', 'Mistral', 'Gemma', 'Gemini-1.5-Pro (API)'))

if user_question:
    Question = (user_question)
    if option == "Flan-T5-base(Fine-Tuned)":
        Model = Rag.Flan_T5_base(user_question=Question)
    if option == "Mistral":
        Model = Rag.Mistral(user_question=Question)
    if option == "Gemma":
        Model = Rag.Gemma(user_question=Question)
    if option == "Gemini-1.5-Pro (API)":
        Model = Rag.GeminiPro(user_question=Question)
    info_placeholder = st.empty()
    info_placeholder.info("Generating Output")
    response = Model.generate()
    info_placeholder.empty()
    if option == "Gemini-1.5-Pro (API)":
        st.write("Reply :", response["output_text"])
    else:
        st.write("Reply :", response)