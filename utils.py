# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import  ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader


pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME")


# Extract Information from PDF file
def get_pdf_text(uploaded_file):
    if uploaded_file:
        temp_file = uploaded_file.name
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

        document = PyPDFLoader(temp_file)
        documents = document.load_and_split()
        return documents


#Create embeddings instance
def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Push data to pinecone
def push_to_pinecone(docs, embeddings):
    index_name = pinecone_index_name
    store = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    return store


def pull_from_pinecone(embeddings):
   index_name = pinecone_index_name
   vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
   return vectorstore


# make the model of the chat_history and context in order to retrieve rrelevant document
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# final lap which instruct the llm to anwers the user query based on the history, relevant docs and prompt
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
      ("""system", "Answer the user's questions based on only the below context:\n\n{context}

       Keep your answer ground in the facts of the context. If the answer is not contained in the context, say \"I don't know\".
       """),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

	# making sure that the prompt fits into the context window of the llm
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_pdf_data(prompt):
    retriever_chain = get_context_retriever_chain(st.session_state.store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": prompt
    })
    return response['answer']