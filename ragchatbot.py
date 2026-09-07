import pdfplumber
import streamlit as st #learn how to streamlit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter #LEARN LANGCHAIN
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = "" #input API Key (api key should be sotred in another environment or outside place to keep security)

#first step is to read the pdf file by creating ui
st.header("My First RagChatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your PDF file and start asking your questions", type="pdf")

#next step is to read the PDF file and chunk it out
#Extract contents from the PDF and chunk it
if file is not None:
    #extract the text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    #st.write(text)

    #Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        #separator array tell us that anytime any of the characters in the array are found, we can chunk out that section
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        #overlap represents keeping an overlap of characters in each chunk so that there are no abrupt stops in explanation
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings (try creating your own vector embeddings)
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        openai_api_key = OPENAI_API_KEY
    )

    #store embeddings in vector db, frist argument determines what do you need embeddings for which is the chunks
    #the next argument determined how are you going to be doing your embeddings which is done through the OpenAI model
    vector_store = FAISS.from_texts(chunks , embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #generate answer
    #question -> embeddings -> similarity search -> results to LLM -> end response (Chain of events)
    #retriever gets the content which is the formatted through format_docs
    #the information is then put into a prompt with understandable input which gets fed into the LLM
    #StrOuputParser removes all Metadata and shows valuable data to the user

    #formts the content cleanly
    def format_docs(docs):
        return "\n\n".join([docs.page_content for doc in docs])

    #retriever variable (mmr is searching technique, k 4 means return the 4 closest matches)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    #define the LLM and prompts
    #heavier tasks would require stronger models
    #temperature determines how creative answers are. Answer should be deterministic so closer to 0
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=OPENAI_API_KEY
    )

    #provide the prompts
    #define template for prompt behavior
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below. \n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge. \n"
         "5. Summarize long information, ideally in bullets where needed. \n"
         "5. If the information is not in the context, say so politely. \n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)