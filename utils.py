from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.llms.openai import OpenAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.llms import HuggingFaceHub
# from langchain_pinecone import PineconeVectorStore
import os
import time
from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME")



# Extract Information from PDF file
def get_pdf_text(uploaded_file):
    if uploaded_file:
        temp_file = uploaded_file.name
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_file)
        documents = loader.load_and_split()
    return documents


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#old
# def push_to_pinecone(docs, embeddings):
#     # Pinecone(api_key=pinecone_api_key)
#     index_name = pinecone_index_name
#     index = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
#     return index


#Function to help us get relavant documents from vector store - based on user input
def similarity_search(docsearch,query,k,unique_id):
     # similarity_search_with_score returns score % assign to each seacrh doc
    time.sleep(20)
    matched_docs = docsearch.similarity_search_with_score(query, int(k), {"unique_id":unique_id})
    return matched_docs


# Summarise doc
# def get_summary(current_doc):
#     llm = OpenAI(temperature=0)
#     #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
#     chain = load_summarize_chain(llm, chain_type="map_reduce")
#     summary = chain.run([current_doc])
#     return summary
