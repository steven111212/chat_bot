import os
os.environ["OPENAI_API_KEY"] = "Your-api-key"
os.environ['USER_AGENT'] = 'myagent'
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory




class Dataload:
    def __init__(self, route):
        
        self.route = route


    def chunk_embedding(self):

        if self.route[-3:] == 'pdf':
            raw_pdf_elements = partition_pdf(
            filename= self.route,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf= False,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure= False,
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            # Chunking params to aggregate text blocks
            # Attempt to create a new chunk 3800 chars
            # Attempt to keep chunks > 2000 chars
            # Hard max on chunks
            max_characters=800,
            new_after_n_chars=600,
            combine_text_under_n_chars=300,
        )
            all_splits = []
            for e in raw_pdf_elements:
                doc = Document(
                page_content = e.text,
                metadata = {
                    'type': 'text'
                }
            ) 
                all_splits.append(doc)
            
        elif self.route[-3:] == 'txt':

            all_splits = []
            with open(self.route, 'r', encoding='utf-8') as file:
                text = file.read()
            text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_text(text)
            for e in texts:
                doc = Document(
                page_content = e,
                metadata = {
                    'type': 'text'
                }
            ) 
                all_splits.append(doc)

        elif self.route[:4] == 'http':
            loader = WebBaseLoader(self.route)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_documents(data)

        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(k=4)

    
    def chat(self, input_text):

        docs = self.retriever.invoke(input_text)

        chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

        self.demo_ephemeral_chat_history = ChatMessageHistory()

        self.demo_ephemeral_chat_history.add_user_message(input_text)

        response = document_chain.invoke(
            {
                "messages": self.demo_ephemeral_chat_history.messages,
                "context": docs,
            }
        )

        print(response)
        self.demo_ephemeral_chat_history.add_ai_message(response)
    

    

# test = Dataload("https://www.gvm.com.tw/article/114355")
# retriever = test.chunk_embedding()
# while True:
#     input_text = input()
#     test.gogo(input_text)