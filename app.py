import streamlit as st
from dotenv import load_dotenv
from utils import *
from langchain_core.messages import AIMessage, HumanMessage
from typing import Union

# Load environment variables
load_dotenv()

def main() -> None:
    """Main function to run the Streamlit application."""

    st.set_page_config(
        page_title="Bot",
        page_icon='🤖💬',
        layout='centered',
        initial_sidebar_state='expanded'
    )
    st.markdown("<h3 style='text-align: center;'>How can I assist you? </h3>", unsafe_allow_html=True)

    # Creating Session State Variable
    if 'store' not in st.session_state:
        st.session_state['store'] = None

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            AIMessage(content="Hello, I am a bot. How can I brighten your day?")
        ]

    # Upload file
    with st.sidebar:
        st.title("📄💬➡️🔍")
        pdf_file = st.file_uploader("Upload files here, only PDF files allowed", type=["pdf"])

    # Check if file is present
    if pdf_file:
        try:
            with st.spinner("Uploading file..."):
                # Create embeddings instance
                embeddings = create_embeddings()

                if os.path.exists(pdf_file.name):
                    st.session_state['store'] = pull_from_pinecone(embeddings)
                else:
                     # Create a list of documents from uploaded PDF files
                    docs = get_pdf_text(pdf_file)

                    # Push data to Pinecone
                    st.session_state['store'] = push_to_pinecone(docs, embeddings)

            # Enter prompt
            prompt: Union[str, None] = st.chat_input("Enter a prompt here")

            if prompt:
                response: str = get_pdf_data(prompt)

                # Append user prompt and response to chat history
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=response))

            # Display conversation history
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Upload a PDF file to get started.")


if __name__ == '__main__':
    main()
