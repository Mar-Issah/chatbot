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


def get_pdf_text(uploaded_file):
    """
    Extract text from a PDF file.

    Args:
        uploaded_file (BytesIO): PDF file uploaded by the user.

    Returns:
        list: List of extracted text documents.
    """
    if uploaded_file:
        temp_file = uploaded_file.name
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

        document = PyPDFLoader(temp_file)
        documents = document.load_and_split()
        return documents
    else:
        raise ValueError("No PDF file provided.")


def create_embeddings_load_data():
    """Create an instance of OpenAI embeddings."""
    return OpenAIEmbeddings()



def push_to_pinecone(docs, embeddings):
    """
    Push documents to Pinecone vector store.

    Args:
        docs (list): List of documents.
        embeddings: Embeddings instance.
        index_name (str): Name of the Pinecone index.

    Returns:
        PineconeVectorStore: Vector store.
    """
    vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return vectorstore


# Pull from pinecone
def pull_from_pinecone(embeddings):
    """
    Pull documents from Pinecone vector store.

    Args:
        embeddings: Embeddings instance.
        index_name (str): Name of the Pinecone index.

    Returns:
        PineconeVectorStore: Vector store.
    """
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
    return vectorstore


def get_context_retriever_chain(vector_store):
    """Create a retriever chain based on context."""
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# Final lap which instructs the llm to anwers query based on the history, relevant docs
def get_conversational_rag_chain(retriever_chain):
    """Create a RAG chain for conversational retrieval."""
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
    """
    Get data from PDF based on conversation context.

    Args:
        prompt (str): User prompt.
        retriever_chain: Retriever chain.
        chat_history (str): Chat history.

    Returns:
        str: Response to the user's query.
    """
    retriever_chain = get_context_retriever_chain(st.session_state.store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": prompt
    })
    return response['answer']
