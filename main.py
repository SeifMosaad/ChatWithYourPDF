import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
if __name__ == "__main__":
    print("HI")
    # pdf_path = "D:\\LangChain\\introduction_to_vectordb\\p2098-qiu.pdf"
    # loader = PyPDFLoader(file_path=pdf_path)
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=30, separator="\n"
    # )
    # docs = text_splitter.split_documents(documents=documents)
    #
    embeddings = OpenAIEmbeddings()
    # vector_store = FAISS.from_documents(docs, embeddings)
    # vector_store.save_local("faiss_index_react")
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever = new_vectorstore.as_retriever()
    )
    res = qa.run("What is the main problem, and what are the contributions?")
    print(res)

