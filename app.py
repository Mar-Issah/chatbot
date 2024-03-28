import streamlit as st
from dotenv import load_dotenv
from utils import *
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Bot",
        page_icon='🤖💬',
        layout='centered',
        initial_sidebar_state='expanded'
    )
    st.markdown("<h3 style='text-align: center;'>How can I assist you? </h3>", unsafe_allow_html=True)

    # Creating Session State Variable
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            AIMessage(content="Hello, I am a bot. How can I brighten your day?")
        ]

    # Upload file
    with st.sidebar:
        st.title("📄💬➡️🔍")
        pdf_file = st.file_uploader("Upload files here, only PDF files allowed", type=["pdf"])

        # check if file is present
    if pdf_file is None or pdf_file == "":
        st.warning("Please upload a file!")
    else:
        try:
            # Create a list of documents from uploaded PDF files
            docs = get_pdf_text(pdf_file)
            st.write(docs)

            # Create embeddings instance
            embeddings = create_embeddings_load_data()

            # Push data to Pinecone
            # docsearch = push_to_pinecone(docs, embeddings)

            # response_container = st.container()

            prompt = st.chat_input("Enter a prompt here")

            # Uncomment the following lines if you want to append user's message and AI's response
            # if prompt:
            #     st.session_state['messages'].append(prompt)
            #     model_response = get_response(prompt)
            #     st.session_state['messages'].append(model_response)

            # Finally display the user message and AI message
            # with response_container:
            #     for i in range(len(st.session_state['messages'])):
            #         if (i % 2) == 0:
            #             message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
            #         else:
            #             message(st.session_state['messages'][i], key=str(i) + '_AI')

        except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Invoking main function
if __name__ == '__main__':
    main()
