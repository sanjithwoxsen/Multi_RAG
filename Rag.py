from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-base")
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import datetime

#Initial setup
Time = datetime.datetime.now() # Output: Current date and time in YYYY-MM-DD HH:MM:SS.SSSSSS format
load_dotenv() #Loading Environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print("Connection established with Google")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def pdf_to_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        Reader = PdfReader(pdf)
        for page in Reader.pages:
            text += page.extract_text()
    print("Text extracted")
    return text

def text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=3000)
    chunks = text_splitter.split_text(text)
    print("Broken Into Chunks")
    return chunks

def vectorstore_faiss(text_chunks):
    vectors_store = FAISS.from_texts(text_chunks , embeddings)
    vectors_store.save_local("faiss_index")
    print("Vector Embeddings saved")

def vectorstore_chroma(text_chunks):
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        collection_name="rag-chroma",
        embedding=embeddings, persist_directory="./chroma_db"
    )
def vectorstore_universal(text_chunks):
    vectors_store = FAISS.from_texts(text_chunks, embeddings)
    vectors_store.save_local("faiss_index")
    print("Vector Embeddings saved")

    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        collection_name="rag-chroma",
        embedding=embeddings, persist_directory="./chroma_db"
    )


class GeminiPro:
    def __init__(self,user_question):
        self.user_question = user_question

    def conversational_chain_gemini(self):
        prompt_template = """Answer as correct as possible with complete detail with the provided context.\n\n
        Context:\n{context}\n
        Question:\n{question}?\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.9)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def retrive_faiss(self):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(self.user_question)
        pdf_docs = [doc.page_content for doc in docs]
        # Outputting the source documents
        filename = "log.txt"
        with open(filename, "w") as file:
            file.write(f"Question:{self.user_question} Time:{Time}")
            for i, pdf_doc in enumerate(pdf_docs, start=1):
                file.write(f"Document {i}:")
                file.write(pdf_doc)
        return docs

    def get_response(self,docs):
        chain = self.conversational_chain_gemini()
        response = chain({"input_documents": docs, "question": self.user_question}, return_only_outputs=True)
        return response

    def generate(self):
        input_doc = self.retrive_faiss()
        response = self.get_response(input_doc)
        return response


class Mistral:
    def __init__(self,user_question):
        self.user_question = user_question

    def retrive_chroma(self):
        retriever = Chroma(persist_directory="./chroma_db", collection_name="rag-chroma",
                           embedding_function=embeddings).as_retriever()
        return retriever
    def prompt(self):
        rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        return rag_prompt
    def generate(self):
        retriever = self.retrive_chroma()
        rag_prompt = self.prompt()
        model_local = ChatOllama(model="mistral")
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | model_local
                | StrOutputParser()
        )
        response = rag_chain.invoke(self.user_question)
        return response
class Gemma:
    def __init__(self,user_question):
        self.user_question = user_question

    def retrive_chroma(self):
        retriever = Chroma(persist_directory="./chroma_db", collection_name="rag-chroma",
                           embedding_function=embeddings).as_retriever()
        return retriever
    def prompt(self):
        rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        return rag_prompt
    def generate(self):
        retriever = self.retrive_chroma()
        rag_prompt = self.prompt()
        model_local = ChatOllama(model="gemma")
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | model_local
                | StrOutputParser()
        )
        response = rag_chain.invoke(self.user_question)
        return response

class Flan_T5_base:
    def __init__(self,user_question):
        self.user_question =user_question
        self.tokenizer = AutoTokenizer.from_pretrained("VIN-IT/results_new")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("VIN-IT/results_new")

    def input_text(self):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(self.user_question)
        pdf_docs = [doc.page_content for doc in docs]
        self.context = "\n".join(pdf_docs)

    def generate(self):
        self.input_text()
        input_text = self.context.replace("\n", " ")
        input_text = "context: " + input_text + " question: " + self.user_question + " </s>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        outputs = self.model.generate(input_ids, max_length=2000, num_beams=4, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response





