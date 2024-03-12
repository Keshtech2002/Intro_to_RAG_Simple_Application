# Importing Libraries to use
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from getpass import getpass
import os
import pathlib
import textwrap
import pickle

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Getting API key from another environment file
from dotenv import load_dotenv, find_dotenv
load_dotenv('api_keys.env')
# _ = load_dotenv(find_dotenv()) # read
api_key = os.environ["GEMINI_API_KEY"]


# api_key = getpass(api_key)

# Reading embedding model into a variable using pickle
with open('embedding.pkl', 'rb') as file:
    embedding = pickle.load(file)


# Large model to be used
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,  google_api_key=api_key, convert_system_message_to_human=True)


persist_directory = 'chroma_db'


# Creating a template for the RAG
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Always say "Thank you for the question" after your answers
Question: {question}
Context: {context}
Answer:
"""


def load_db(file, k, embedding=embedding, persist_directory=persist_directory, llm=llm):
    # load documents
    # loader = PyPDFLoader(file)
    loader = TextLoader(file, encoding='utf-8')
    documents = loader.load()
    
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # create vector database from data
    # db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    
    # define retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Passing the template into the prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # create a chatbot chain. Memory is managed externally.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_db("Relunctant_to_go_but_Victory_came_back.txt", k=3)
response = rag_chain.invoke("Where is badr located?")
print(response)


rag_chain = load_db("Relunctant_to_go_but_Victory_came_back.txt", k=3)
response = rag_chain.invoke("Who is Mus'ab Keshinro?")
print(response)


rag_chain = load_db("Relunctant_to_go_but_Victory_came_back.txt", k=3)
response = rag_chain.invoke("Who is Obafemi Awolowo")
print(response)
#

# if _name_ == "_main_":
#     print(qa({"question": "When did tinubu start to head the APC"})["answer"])