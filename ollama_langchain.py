## langchain
from langchain.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader

class OllamaLangchain:
    def generate_vectorstore(self, data, ollama_server, model_tag):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        all_splits = text_splitter.split_documents(data)
        oembed = OllamaEmbeddings(base_url=ollama_server, model=model_tag)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
        return vectorstore

    def open_url(self, url, ollama_server, model_tag):
        loader = WebBaseLoader(url)
        data = loader.load()
        return self.generate_vectorstore(data, ollama_server, model_tag)

    def open_pdf(self, path, ollama_server, model_tag):
        loader = PyPDFium2Loader(path)
        data = loader.load_and_split()
        return self.generate_vectorstore(data, ollama_server, model_tag)

    def query(self, vectorstore, prompt, ollama_server, model_tag):
        ollama = Ollama(base_url=ollama_server,model=model_tag)
        docs = vectorstore.similarity_search(prompt)
        qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
        return qachain({"query": prompt})
