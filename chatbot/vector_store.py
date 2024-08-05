from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    
    def create_from_documents(documents):
        return Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())