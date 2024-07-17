import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # chat with our vector store
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.llms import ollama # local use llama
from langchain_community.chat_models import ChatOllama  # using the chatmodel ollam
from langchain_core.prompts import ChatPromptTemplate

# from langchain.embeddings import OpenAiEmbeddings # openai embeddings
# from langchain_community.llms import OpenAI # to use chatgpt
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from HtmlTemplates import css, bot_template, user_template  # import them from the html
from FlagEmbedding import BGEM3FlagModel
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


def get_pdf_text(pdf_documents):
    text = ""  # initialize the text

    # loop through the pdf in pdf_docs after that append the texts in the pages to text:
    for pdf in pdf_documents:  # loop through our pdfs
        pdf_reader = PdfReader(pdf)  # initialize pdfreader object for each pdf
        for page in pdf_reader.pages:  # loop through the pages for each pdf
            text += page.extract_text()  # extract the text and add it to our string
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )  # a new istance of charactersplitter, the arg chunk_size= how many char you will take and overlap is to protect us from breaking the meaning of a chunk so it will always take some characters back before starting a new chunk
    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAiEmbeddings()  # embeddings using openai paid``
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl",
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    embeddings = model.encode(text_chunks, batch_size=12, max_length=8192)[
        "dense_vecs"
    ]  #  change the embeddings
    # # chroma db
    # vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings)
    # vectorstore = FAISS.from_texts(
    #     texts=text_chunks, embedding=embeddings
    # )  # it takes the embeddings and the chunk of texts as arg  # vector store will get the data from the text (chunk of texts)
    # embeddings = OpenAiEmbeddings()  # embeddings using openai paid
    # embeddings = OllamaEmbeddings(model_name="llama:70b")
    vectorstore = Chroma.from_texts(
        texts=text_chunks, embedding=lambda texts: embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = Ollama(model="llama3", verbose=True)
    llm = ChatGroq(model="llama3-70b-8192")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )  # instatiate the memorybuffer
    conversation_chain = (
        ConversationalRetrievalChain.from_llm(  # it takes these arguments
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,  # configuring the db as retriever
        )
    )
    return conversation_chain


def handle_userinput(user_question):
    # we will use the same conversation vari
    response = st.session_state.conversation.invoke(
        {"question": user_question}
    )  # as the conversation contains all the v db and memory it will remember the inputs
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # take the odd numbers of the history
            st.write(
                user_template.replace(
                    "{{MSG}}", message.content
                ),  # to get only the human message from the message
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with your PDFs", page_icon=":scroll:"
    )  # customize the streamlit interface
    # we should always add the css on top just like on a website
    st.write(css, unsafe_allow_html=True)
    if (
        "conversation" not in st.session_state
    ):  # if conversation is not initialized it will be set to None
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # always remember to initialize it
    st.header("Chat with your PDFs :books:")  # main header
    user_question = st.text_input(
        "Ask a question about your document: "
    )  # the input box
    if user_question:
        handle_userinput(
            user_question
        )  # user submits the question than the input is handled using llm

    # st.write(
    #     user_template.replace("{{MSG}}", "Hello Robot"),
    #     unsafe_allow_html=True,  # figure out why it does not show for now hard encode it in the html template
    # )  # unsafe_allow_ supposed to show html template and parse it
    # st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True) # we do not need it here as we used in the handle func

    with st.sidebar:  # set up the sidebar
        st.subheader("Your documents")
        pdf_documents = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )  # to upload the pdf, allow multiple files
        if st.button("Process"):
            with st.spinner(
                "Processing"
            ):  # add a loading screen user friendly for the user
                # get the pdf raw text
                raw_text = get_pdf_text(
                    pdf_documents
                )  # it will take all the content from the pdfs
                # st.write(raw_text) this statment is to check how the pdf are displayed as a chunk of texts before using the split function

                # split it into chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                print(type(text_chunks))
                # create the vector store
                # first using openAI embeddings:
                vectorstore = get_vectorstore(text_chunks)
                # conversation chain
                # user streamlit, involves reinitializing some var after each mdoficiation in the code or something else so to make sure that a var is not reinitialized we use st.session_state
                st.session_state.conversation = get_conversation_chain(vectorstore)
    # session stat will allow us to use the var outside of the scope  and also we need to initialize it , when the app is open st realoads the code
    #  we can : do this :
    # st.session_state.conversation

    # we need to display messages using an html template or you can use streamlit chat


if __name__ == "__main__":
    main()
