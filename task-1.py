pip install langchain faiss-cpu PyPDF2 tiktoken openai beautifulsoup4 requests






import os
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Task 1: Chat with PDFs using RAG

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

def extract_text_from_pdfs(pdf_files):
    """Extract text from PDF files"""
    all_documents = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())
    return all_documents

def process_and_store_embeddings(documents):
    """Process documents: split into chunks, embed, and store in vector DB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def answer_query(vector_db, query):
    """Answer user query using RAG pipeline."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    response = qa_chain.run(query)
    return response

def main_task_1(pdf_files, query):
    # Step 1: Extract text from PDFs
    documents = extract_text_from_pdfs(pdf_files)
    
    # Step 2: Process and store embeddings
    vector_db = process_and_store_embeddings(documents)
    
    # Step 3: Answer query
    response = answer_query(vector_db, query)
    print("Response:", response)

# Example Usage for Task 1
pdf_files = ["example1.pdf", "example2.pdf"]  # Replace with your PDF file paths
query = "What is the unemployment information based on type of degree?"
main_task_1(pdf_files, query)
